import os
import sys
os.chdir(snakemake.params.proj_dir)
# %%
from datetime import datetime
import numpy as np
import json
from sklearn.externals import joblib
from src.tools.PairsIndex import PairsIndex
from src.tools import cv_inds
import tables as tb

# %% snakemake params:
print("Starting to predict!")
targets=sorted([line.strip() for line in open(
        snakemake.input.targets, "r").readlines()])
model = snakemake.wildcards.model
partial_index = snakemake.wildcards.partial_index
current_range = snakemake.params.current_range
model_dir = snakemake.params.model_dir
all_cv = snakemake.params.all_cv

out_path = snakemake.output[0]

ind = PairsIndex.from_file(snakemake.params.genes_list)

params = json.load(open("config/gen_params.json"))

# subprocess.run(['rm', '-f', out])
#%% load data
print(f"Loading..", file=sys.stderr)
st = datetime.now()
inds = list(range(current_range['st'], current_range['en']))
df = cv_inds.single_CV_fold({"dat": (inds, inds)}, feats=[], params=params, ind=ind)['dat'].drop(columns=["target"])
print(f"Loaded! elapsed: {(datetime.now() - st).total_seconds():.2f}s",
        file=sys.stderr)

# %% predict
print("Predicting...")
st = datetime.now()
res = df.loc[:,['ind']].copy()
for target in targets:
    print("Predicting", target)
    clf, _ = joblib.load(model_dir+f"/{target}/{target}_est.pkl")
    if all_cv:
        print("Using All CVs")
        for i in range(len(clf)):
            probas = clf[i].predict_proba(df.drop(columns='ind'))[:,1]
            res[target+f"_CV{i}"] = probas
    else:
        print("Using mean of CVs")
        res[target]=0.0
        for i in range(len(clf)):
            probas = clf[i].predict_proba(df.drop(columns='ind'))[:,1]
            res[target] += probas
        res[target] = res[target]/len(clf)

print("Finshed! elapsed: {} seconds".format(
    (datetime.now() - st).total_seconds()))
#%%
res['ind'] = res['ind'].astype(np.uint32)
res.iloc[:,1:] = res.iloc[:,1:].astype(np.float32)
res.set_index('ind', inplace=True)

print(res.head())
# %% save to hdf
print("Writing to HDF")

out = tb.open_file(out_path, "w", title="Data")
group = out.create_group("/", "data")
table = out.create_table(group, "data",obj=res.to_records(),
                         title="Probas", expectedrows=res.shape[0])
table.flush()

# %%
table.attrs.st_ind = res.index[0]
table.attrs.en_ind = res.index[-1]
print("Table:\n", table.__str__())
out.close()
