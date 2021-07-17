# %% imports:
import argparse
import json
import os
from datetime import datetime
from multiprocessing import cpu_count

import pandas as pd
import tables as tb
from lightgbm import LGBMClassifier
from sklearn.externals import joblib

from src.external.AdaSample import AdaSample
from src.external.baggingPU import BaggingClassifierPU
from src.tools import cv_inds
from src.tools.PairsIndex import PairsIndex

import argparse
parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('-w', '--which', dest='which',
                    help='Choose which label to train: in_Reactome, inReactome_Path (case insensitive)')
parser.add_argument('-t', '--test', dest='test', action='store_true',
                    help='For testing purposes, will train on only 5e4 rows')
parser.add_argument('--PU', dest='PU',
                    help='Type of PU framework - PUBag or AdaSample or None')
parser.set_defaults(test=False)
args = parser.parse_args() #['-w','in_Reactome', '--PU', 'None'])

# %% Load data

params = json.load(open("config/gen_params.json", 'r'))
feats = params['feats']
PM_splits = ["train", "test", "C1", "C2", "C3"]

i_all = 0
# %% LOAD
ind = PairsIndex.from_file(params['index_path'])

df_para = pd.DataFrame(tb.open_file(params["blastp_hdf"], "r").root["/data"].read())

print("Data loaded")

# %% set label for testing from argparse / snakemake:
wh = args.which
wh = wh.lower()

PU = args.PU

metrics = ['binary_logloss', 'auc']

if wh == 'in_reactome':
    dat = pd.read_csv("data/inGmt.csv")
    dat['ind'] = ind.get_ind_pair([x for x in zip(dat.gene_a, dat.gene_b)])
    targets = ["in_Complex_reactome", "in_Reactome"]
    name_path = "in_Reactome"
elif wh == 'inreactome_path':
    dat = pd.read_csv("data/inReactome_path.csv")
    dat['ind'] = ind.get_ind_pair([x for x in zip(dat.gene_a, dat.gene_b)])
    targets = dat.columns.drop(["gene_a", "gene_b", "ind"]).tolist()
    name_path = "inReactome_Path"
elif wh == 'ingo_path':
    dat = pd.read_csv("data/inGO_path.csv")
    dat['ind'] = ind.get_ind_pair([x for x in zip(dat.gene_a, dat.gene_b)])
    targets = dat.columns.drop(["gene_a", "gene_b", "ind"]).tolist()
    name_path = "inGO_Path"
else:
    raise Exception(wh + " is not a suitable label for training, see -h")

fit_params = json.load(open("config/model_params.json", 'r'))

# %% for testing:
if args.test:
    dat = dat.sample(int(2e4))
    print("Using only 2e4 rows for testing purposes")

# Params:
path = f"models/{name_path}"
os.makedirs(path, exist_ok=True)

print("cpus:", cpu_count())
n_cpu = cpu_count()

# %% Training the model:
st = datetime.now()
print("Training""...")

clfs = {}
for target in sorted(targets):
    print(target + "\n==========================\n")
    curpath = path + f"/{target}"
    os.makedirs(curpath, exist_ok=True)

    ## Calculate CV folds
    cv_folds = cv_inds.CV(dat, target, ind, feats=feats,
                          params=params,
                          CV_path=curpath + f"/splits",
                          splits_save_dir=curpath + f"/splits")

    ## Get splits in memory (and saved as hdf)
    splits = []
    for i in range(params['n_splits']):
        splits.append({})
        for k in PM_splits:
            x, y, colnames_X = cv_inds.get_Xy(cv_folds.cv_folds[i][k])
            splits[i][k] = {"X": x, "y": y}

    ## Get target specific fit params
    if target in fit_params.keys():
        fit_params_tmp = fit_params[target]
    else:
        fit_params_tmp = fit_params.copy()

    ## Split fit params by object
    LGBM_params = dict(**{_k.replace("LGBM_", ""): _v
                          for _k, _v in fit_params_tmp.items() if _k.startswith("LGBM_")})
    AdaSample_params = dict(**{_k.replace("AdaSample_", ""): _v
                               for _k, _v in fit_params_tmp.items() if _k.startswith("AdaSample_")})
    PUBag_params = dict(**{_k.replace("PUBag_", ""): _v
                           for _k, _v in fit_params_tmp.items() if _k.startswith("PUBag_")})

    # add general params
    LGBM_params.update({'verbose': -1, "metrics": metrics, "n_jobs": -1})

    evals_result = {}
    clfs[target] = []
    for l, split in enumerate(splits):
        if PU == "AdaSample":
            clfs[target].append(AdaSample(LGBMClassifier(**LGBM_params), **AdaSample_params))
            clfs[target][l].fit(split['train']['X'], split['train']['y'],
                                eval_set=[(split[k]['X'], split[k]['y']) for k in PM_splits],
                                eval_names=PM_splits,
                                verbose=100)
            evals_result[l] = [est.evals_result_ for est in clfs[target][l].estimators_]
        elif PU == "PUBag":
            clfs[target].append(BaggingClassifierPU(LGBMClassifier(**LGBM_params), **PUBag_params))
            clfs[target][l].fit(split['train']['X'], split['train']['y'],
                                fit_params=dict(eval_set=[(split[k]['X'], split[k]['y']) for k in PM_splits],
                                                eval_names=PM_splits,
                                                verbose=100))
            evals_result[l] = [est.evals_result_ for est in clfs[target][l].estimators_]
        elif PU == "None":
            clfs[target].append(LGBMClassifier(**LGBM_params))
            clfs[target][l].fit(split['train']['X'], split['train']['y'],
                                eval_set=[(split[k]['X'], split[k]['y']) for k in PM_splits],
                                eval_names=PM_splits,
                                verbose=100)
            evals_result[l] = clfs[target][l].evals_result_

    ## save eval_results as json:
    json.dump(evals_result, open(curpath + "/eval_results.json", "w"))

    ## save individual estimator
    joblib.dump((clfs[target], feats), curpath + f"/{target}_est.pkl")

with open(path + "/targets", "w") as f:
    f.write("\n".join(targets))

print("Finished!, elapsed: {} seconds\n====================================\n". \
      format((datetime.now() - st).total_seconds()))

