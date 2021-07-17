# %% DEPS
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')

import seaborn as sns
# sns.set_palette("tab20")

import matplotlib.pyplot as plt

from src.tools.PairsIndex import PairsIndex
from src.tools import cv_inds
from src.visualization import plot_metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from lightgbm import LGBMClassifier
import lightgbm

from sklearn.metrics import roc_auc_score, average_precision_score
import tables as tb
import os
import gc
import json
import argparse

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--which")
args = parser.parse_args()

# %%
figpath = "reports/figures/01_comparing_classifiers"
os.makedirs(figpath, exist_ok=True)

params = json.load(open("config/gen_params.json", 'r'))
params['n_splits'] = 4

PM_splits = ["train", "test", "C1", "C2", "C3"]
PARA_STAT = ['unfilt', 'para_filt']

out = open(figpath + f"/res_{args.which}.csv", 'w', buffering=1)

i_all = 0
# %% LOAD
ind = PairsIndex.from_file(params["index_path"])

df_para = pd.DataFrame(tb.open_file(params["blastp_hdf"], "r").root["/data"].read())

# %% training func:
clfs = {'LGBM': LGBMClassifier(n_estimators=50, subsample=0.5, colsample_bytree=0.5,
                               boosting_type='rf', bagging_freq=1),
        "LR": SGDClassifier(loss='log', max_iter=1000, tol=1e-3),
        'DT': DecisionTreeClassifier(max_depth=6),
        "NB": GaussianNB()}


def calc_splits(cv_folds, config):
    splits = []
    for i, cv_fold in enumerate(cv_folds.cv_folds):
        tmp = cv_fold.inds_dict
        splits.append({})
        for pm_split in PM_splits:
            splits[i][pm_split] = cv_inds.get_cor_clades(inds=tmp[pm_split][0], ind=ind,
                                                         **config)
            splits[i][pm_split]['target'] = tmp[pm_split][1]
    return splits


def train_clfs(cv_folds, target, clfs, para_status, scaler=True):
    print("\n", target, "\n========================\n", sep="")
    clfs_target = {}
    for k in clfs.keys():
        clfs_target[k] = []
        for j, split in enumerate(cv_folds.cv_folds):
            if scaler:
                clfs_target[k].append(Pipeline([
                    ('scale', StandardScaler()),
                    (k, clone(clfs[k], safe=False))]))
            else:
                clfs_target[k].append(Pipeline([(k, clone(clfs[k]))]))

            print("Training {}, CV {}...\n______________\n".format(k, j))
            # print(clfs_target[k][j])
            train = split["train"]

            if para_status == "para_filt":
                train_tmp = train.loc[~train['ind'].isin(df_para['ind']), :]
            else:
                train_tmp = train

            X = train_tmp.drop(columns=["ind", "target"])
            y = train_tmp['target']
            clfs_target[k][j].fit(X, y)

    return clfs_target, cv_folds


def _init_summary(splits=PM_splits, para_statuses=PARA_STAT):
    res_dict = {}
    for para_status in para_statuses:
        res_dict[para_status] = {}
        for split in splits:
            res_dict[para_status][split] = [0] * params['n_splits']
    return res_dict


mean_fpr = np.linspace(0, 0.1, 50)


def train_and_vis(dat, targets, clfs,  figpath_cur, clade_comp=["Eukaryota"], size_w=5.5, size_h=5.5,
                  clf_size_factor=1):
    global i_all
    clf_names = list(clfs.keys()) + ["NPP"]
    all_clfs = {}

    figs = clf_names
    fig_ROC = {}
    axs_ROC = {}
    fig_pROC = {}
    axs_pROC = {}
    fig_PR = {}
    axs_PR = {}

    fig_iter = [[fig_ROC, fig_pROC, fig_PR], [axs_ROC, axs_pROC, axs_PR],
                ["ROC", "pROC", "PR"]]
    for x in figs:
        for fig, axs, name in zip(*fig_iter):
            fig[x], axs[x] = plt.subplots(len(PARA_STAT), len(PM_splits),
                                          figsize=(size_w * clf_size_factor * len(PM_splits),
                                                   size_h * clf_size_factor * len(PARA_STAT)))
            fig[x].tight_layout(rect=[0, 0.03, 1, 0.95])
            fig[x].suptitle("{} {}".format(name, x))

    for target in targets:
        run_one_target(all_clfs, axs_PR, axs_ROC, axs_pROC,
                       clade_comp, clf_names, clf_size_factor, clfs, dat, 
                       fig_PR, fig_ROC, fig_iter, fig_pROC, figpath_cur,
                       size_h, size_w,
                       target)

    for x in clf_names:
        for fig, ax, name in zip(*fig_iter):
            fig[x].tight_layout(rect=[0, 0.03, 1, 0.95])
            fig[x].savefig(figpath_cur + "/{}_{}.png".format(name, x), dpi=300)
            fig[x].savefig(figpath_cur + "/{}_{}.svg".format(name, x))
            plt.close(fig[x])
    plt.close("All")

    return fig_ROC, axs_ROC, fig_PR, axs_PR, all_clfs

def plot_importance(clf_target,cv_folds, target, para_status):
    cols = cv_folds.cv_folds[0]['C3'].drop(columns=["ind", "target"]).columns
    for k, v in clf_target.items():
        fig = plt.figure(figsize=(8,12))
        if k == "DT":
            df = pd.Series(v[0].named_steps[k].feature_importances_, index=cols).sort_values()
            df.plot.barh()
        elif k=="LR":
            df = pd.Series(v[0].named_steps[k].coef_[0,:], index=cols).sort_values()
            df.plot.barh()
        elif k== "LGBM":
            df = pd.Series(v[0].named_steps[k].feature_importances_, index=cols).sort_values()
            df.plot.barh()
        else:
            plt.close(fig)
            continue

        plt.gca().set_title(f"Feat. Importance {k}")
        fig.tight_layout()
        fig.savefig(figpath_cur+f"/importance/{target}_{k}_{para_status}.png", dpi=120)

        if k=="DT":
            fig, ax = plt.subplots(figsize=(9,9))
            plot_tree(v[0].named_steps[k], filled=True ,ax = ax, feature_names=cols)
            fig.tight_layout()
            fig.savefig(figpath_cur + f"/importance/{target}_{k}_TREE_{para_status}.pdf")
            plt.close(fig)


def run_one_target(all_clfs,
                   axs_PR, axs_ROC, axs_pROC,
                   clade_comp, clf_names, clf_size_factor,
                   clfs, dat, 
                   fig_PR, fig_ROC, fig_iter, fig_pROC,
                   figpath_cur, size_h, size_w,
                   target):
    global i_all

    #init figures:
    for x in [target, target + "_NPP"]:
        for fig, axs, name in zip(*fig_iter):
            fig[x], axs[x] = plt.subplots(len(PARA_STAT), len(PM_splits),
                                          figsize=(size_w * len(PM_splits),
                                                   size_h * len(PARA_STAT)))

        fig[x].tight_layout(rect=[0, 0.03, 1, 0.95])
        fig[x].suptitle("{} {}".format(name, x))

    #get cross validations
    cv_folds = cv_inds.CV(dat, target, ind, feats=[], params=params, CV_path=figpath_cur)

    all_clfs[target] = {}

    #####NPP COR##########
    params_NPP = params.copy()
    params_NPP['func'] = "cor"
    params_NPP['npp_path'] = "data/NPP.tsv"
    NPP_cv = [cv_inds.single_CV_fold(split.inds_dict, feats=[], params=params_NPP, ind=ind)
              for split in cv_folds.cv_folds]
    for clade in clade_comp:
        print(f'NPP {clade}\n------------------\n')
        tprs = _init_summary()
        aucs = _init_summary()
        p_tprs = _init_summary()
        p_aucs = _init_summary()
        precs = _init_summary()
        aps = _init_summary()
        sizes = _init_summary()

        res = {"target": target, "classifier": f"NPP_{clade}"}
        for l, para_status in enumerate(PARA_STAT):
            print(para_status)
            for i, PM_split in enumerate(PM_splits):
                for j, split in enumerate(NPP_cv):
                    dat_x = split[PM_split]
                    if para_status == "para_filt":
                        dat_tmp = dat_x.loc[~dat_x['ind'].isin(df_para['ind']), :]
                    else:
                        dat_tmp = dat_x

                    X_s = dat_tmp.drop(columns=["ind", "target"])
                    y_s = dat_tmp['target']

                    probas = X_s[f'{clade}']
                    tmp_auc = roc_auc_score(y_s, probas)
                    tmp_ap = average_precision_score(y_s, probas)
                    print("ROC AUC : {:.3f}\n".format(tmp_auc),
                          "Avg. Prec: {:.3f}\n\n".format(tmp_ap), sep="")

                    tprs[para_status][PM_split][j], \
                    aucs[para_status][PM_split][j] = plot_metrics.generate_tprs(y_s, probas)
                    p_tprs[para_status][PM_split][j], \
                    p_aucs[para_status][PM_split][j] = plot_metrics.generate_tprs(y_s, probas, mean_fpr=mean_fpr)
                    precs[para_status][PM_split][j], \
                    aps[para_status][PM_split][j] = plot_metrics.generate_precs(y_s, probas)
                    sizes[para_status][PM_split][j] = X_s.shape[0]
                    res[f"CV{j} AUC {PM_split} ({para_status})"] = tmp_auc
                    res[f"CV{j} AP {PM_split} ({para_status})"] = tmp_auc
                [[res.update({
                    f"mean AUC {PM_split} ({para_status})": np.mean(aucs[para_status][PM_split]),
                    f"std AUC {PM_split} ({para_status})": np.std(aucs[para_status][PM_split]),
                    f"mean pAUC0.1 {PM_split} ({para_status})": np.mean(p_aucs[para_status][PM_split]),
                    f"std pAUC0.1 {PM_split} ({para_status})": np.std(p_aucs[para_status][PM_split]),
                    f"mean AP {PM_split} ({para_status})": np.mean(aps[para_status][PM_split]),
                    f"std AP {PM_split} ({para_status})": np.std(aps[para_status][PM_split])
                }) for PM_split in PM_splits] for para_status in PARA_STAT]

                title = f"{PM_split} ({para_status})" + f"(size: {np.mean(sizes[para_status][PM_split]):.1e})"

                plot_metrics.plot_roc_cv(tprs[para_status][PM_split], aucs[para_status][PM_split],
                                         clade + "_NPP", ax=axs_ROC[target + "_NPP"][l, i],
                                         figsize=(size_w, size_h),
                                         title=title)
                plot_metrics.plot_roc_cv(p_tprs[para_status][PM_split], p_aucs[para_status][PM_split],
                                         clade + "_NPP", ax=axs_pROC[target + "_NPP"][l, i],
                                         figsize=(size_w, size_h), mean_fpr=mean_fpr,
                                         title=title)
                axs_pROC[target + "_NPP"][l, i].set_ylim((0, 0.6))
                plot_metrics.plot_pr_cv(precs[para_status][PM_split], aps[para_status][PM_split],
                                        clade + "_NPP", ax=axs_PR[target + "_NPP"][l, i],
                                        figsize=(size_w, size_h),
                                        title=title)
                if clade == 'Eukaryota':
                    plot_metrics.plot_roc_cv(tprs[para_status][PM_split], aucs[para_status][PM_split],
                                             target, ax=axs_ROC["NPP"][l, i],
                                             figsize=(size_w, size_h),
                                             title=title)
                    plot_metrics.plot_roc_cv(p_tprs[para_status][PM_split], p_aucs[para_status][PM_split],
                                             target, ax=axs_pROC["NPP"][l, i], mean_fpr=mean_fpr,
                                             figsize=(size_w, size_h),
                                             title=title)
                    axs_pROC["NPP"][l, i].set_ylim((0, 0.6))
                    plot_metrics.plot_pr_cv(precs[para_status][PM_split], aps[para_status][PM_split],
                                            target, ax=axs_PR["NPP"][l, i],
                                            figsize=(size_w, size_h),
                                            title=title)
                    plot_metrics.plot_roc_cv(tprs[para_status][PM_split], aucs[para_status][PM_split],
                                             "NPP", ax=axs_ROC[target][l, i],
                                             figsize=(size_w, size_h),
                                             title=title)
                    plot_metrics.plot_roc_cv(p_tprs[para_status][PM_split], p_aucs[para_status][PM_split],
                                             "NPP", ax=axs_pROC[target][l, i], mean_fpr=mean_fpr,
                                             figsize=(size_w, size_h),
                                             title=title)
                    axs_pROC[target][l, i].set_ylim((0, 0.6))
                    plot_metrics.plot_pr_cv(precs[para_status][PM_split], aps[para_status][PM_split],
                                            "NPP", ax=axs_PR[target][l, i],
                                            figsize=(size_w, size_h),
                                            title=title)
        if i_all == 0:
            pd.DataFrame(res, index=[0]). \
                to_csv(out, index=False)
            i_all += 1
        else:
            pd.DataFrame(res, index=[0]). \
                to_csv(out, header=False, index=False)
    for x in [target, target + '_NPP', "NPP"]:
        for fig, ax, name in zip(*fig_iter):
            fig[x].tight_layout(rect=[0, 0.03, 1, 0.95])
            fig[x].savefig(figpath_cur + "/{}_{}.png".format(name, x), dpi=150)
            fig[x].savefig(figpath_cur + "/{}_{}.svg".format(name, x))
            if x == target + '_NPP':
                fig[x].clf()
                plt.close(fig[x])
                del fig[x]

    del NPP_cv
    ##############################
    ######Classifiers############
    tprs = {k: _init_summary() for k in clf_names}
    aucs = {k: _init_summary() for k in clf_names}
    p_tprs = {k: _init_summary() for k in clf_names}
    p_aucs = {k: _init_summary() for k in clf_names}
    precs = {k: _init_summary() for k in clf_names}
    aps = {k: _init_summary() for k in clf_names}
    sizes = {k: _init_summary() for k in clf_names}
    res = {k: {"target": target, "classifier": k} for k in clf_names}
    for l, para_status in enumerate(PARA_STAT):
        print(para_status)
        clfs_target, cv_folds = train_clfs(cv_folds, target, clfs, para_status=para_status, scaler=False)
        all_clfs[target][para_status] = {"clfs": clfs_target, "cv_folds": cv_folds}

        os.makedirs(figpath_cur+"/importance", exist_ok=True)
        plot_importance(clfs_target, cv_folds, target, para_status)

        for k, v in clfs_target.items():
            print(f'{k}\n------------------\n')
            for i, PM_split in enumerate(PM_splits):
                print(PM_split)
                for j, split in enumerate(cv_folds.cv_folds):
                    dat_x = split[PM_split]

                    if para_status == "para_filt":
                        dat_tmp = dat_x.loc[~dat_x['ind'].isin(df_para['ind']), :]
                    else:
                        dat_tmp = dat_x

                    if dat_tmp.shape[0] < 10:
                        continue

                    X_s = dat_tmp.drop(columns=["ind", "target"])
                    y_s = dat_tmp['target']

                    # Calculating probabilities for test
                    # print(v[j])
                    probas = v[j].predict_proba(X_s)[:, 1]
                    tmp_auc = roc_auc_score(y_s, probas)
                    tmp_ap = average_precision_score(y_s, probas)
                    print("ROC AUC : {:.3f}\n".format(tmp_auc),
                          "Avg. Prec: {:.3f}\n\n".format(tmp_ap), sep="")

                    tprs[k][para_status][PM_split][j], \
                    aucs[k][para_status][PM_split][j] = plot_metrics.generate_tprs(y_s, probas)
                    p_tprs[k][para_status][PM_split][j], \
                    p_aucs[k][para_status][PM_split][j] = plot_metrics.generate_tprs(y_s, probas, mean_fpr=mean_fpr)
                    precs[k][para_status][PM_split][j], \
                    aps[k][para_status][PM_split][j] = plot_metrics.generate_precs(y_s, probas)
                    sizes[k][para_status][PM_split][j] = X_s.shape[0]
                    res[k][f"CV{j} AUC {PM_split} ({para_status})"] = tmp_auc
                    res[k][f"CV{j} AP {PM_split} ({para_status})"] = tmp_auc
                [[res[k].update({
                    f"mean AUC {PM_split} ({para_status})": np.mean(aucs[k][para_status][PM_split]),
                    f"std AUC {PM_split} ({para_status})": np.std(aucs[k][para_status][PM_split]),
                    f"mean pAUC0.1 {PM_split} ({para_status})": np.mean(p_aucs[k][para_status][PM_split]),
                    f"std pAUC0.1 {PM_split} ({para_status})": np.std(p_aucs[k][para_status][PM_split]),
                    f"mean AP {PM_split} ({para_status})": np.mean(aps[k][para_status][PM_split]),
                    f"std AP {PM_split} ({para_status})": np.std(aps[k][para_status][PM_split])
                }) for PM_split in PM_splits] for para_status in PARA_STAT]

                _plot_roc_pr(sizes[k][para_status][PM_split],
                             aps[k][para_status][PM_split], aucs[k][para_status][PM_split],
                             precs[k][para_status][PM_split], tprs[k][para_status][PM_split],
                             p_aucs[k][para_status][PM_split], p_tprs[k][para_status][PM_split],
                             fig_ROC, fig_PR, axs_PR, axs_ROC, fig_pROC, axs_pROC,
                             clf_size_factor, i, k, l,
                             size_h, size_w,
                             target, PM_split)
    for x in [target]:
        for fig, ax, name in zip(*fig_iter):
            fig[x].tight_layout(rect=[0, 0.03, 1, 0.95])
            fig[x].savefig(figpath_cur + "/{}_{}.png".format(name, x), dpi=150)
            fig[x].savefig(figpath_cur + "/{}_{}.svg".format(name, x))
            fig[x].clf()
            plt.close(fig[x])
            del fig[x]
    for k in clf_names:
        pd.DataFrame(res[k], index=[0]). \
            to_csv(out, header=False, index=False)
    gc.collect()

def _plot_roc_pr(sizes, aps, aucs, precs, tprs, p_aucs, p_tprs, fig_ROC, fig_PR, axs_PR, axs_ROC, fig_pROC, axs_pROC,
                 clf_size_factor, i, k, l, size_h, size_w, target, PM_split):
    title = f"{PM_split} (size: {np.mean(sizes):.1e})"
    plot_metrics.plot_roc_cv(tprs, aucs, k, ax=axs_ROC[target][l, i],
                             figsize=(size_w, size_h),
                             title=title)
    plot_metrics.plot_roc_cv(tprs, aucs, target, ax=axs_ROC[k][l, i],
                             figsize=(size_w * clf_size_factor, size_h * clf_size_factor),
                             title=title)

    plot_metrics.plot_roc_cv(p_tprs, p_aucs, k, ax=axs_pROC[target][l, i],
                             figsize=(size_w, size_h), mean_fpr=mean_fpr,
                             title=title)
    axs_pROC[target][l, i].set_ylim((0, 0.6))
    plot_metrics.plot_roc_cv(p_tprs, p_aucs, target, ax=axs_pROC[k][l, i], mean_fpr=mean_fpr,
                             figsize=(size_w * clf_size_factor, size_h * clf_size_factor),
                             title=title)
    axs_pROC[k][l, i].set_ylim((0, 0.6))

    plot_metrics.plot_pr_cv(precs, aps, k, ax=axs_PR[target][l, i],
                            figsize=(size_w, size_h),
                            title=title)
    plot_metrics.plot_pr_cv(precs, aps, target, ax=axs_PR[k][l, i],
                            figsize=(size_w * clf_size_factor, size_h * clf_size_factor),
                            title=title)

# %%
#########################################
############ in_Reactome ################
#########################################
if args.which == "in_Reactome":
    figpath_cur = figpath + "/in_Reactome"
    os.makedirs(figpath_cur, exist_ok=True)

    df_pw = pd.read_csv("data/inGmt.csv")
    dat = df_pw#.sample(frac=0.1)
    dat['ind'] = ind.get_ind_pair([x for x in zip(dat.gene_a, dat.gene_b)])

    sns.set_palette("tab10")
    targets = ['in_Reactome', "in_Complex_reactome"]
    train_and_vis(dat, targets, clfs,  figpath_cur,
                  size_h=6, size_w=6,
                  clade_comp=['Eukaryota',
                              'Mammalia',
                              'Chordata',
                              'Alveolata',
                              'Viridiplantae',
                              'Fungi'])
plt.close('all')

# %%
#########################################
######### inGmt Reactome ################
#########################################
if args.which == "inReactome_Path":
    figpath_cur = figpath + "/inReactome_Path"
    os.makedirs(figpath_cur, exist_ok=True)

    df_pw = pd.read_csv("data/inReactome_path.csv")

    dat = df_pw
    dat['ind'] = ind.get_ind_pair([x for x in zip(dat.gene_a, dat.gene_b)])

    sns.set_palette(sns.color_palette("tab10") + sns.color_palette("pastel"))
    targets = sorted(dat.columns.drop(["gene_a", "gene_b", "ind"]).tolist())
    train_and_vis(dat, targets, clfs,  figpath_cur,
                  size_h=6, size_w=6, clf_size_factor=1.7,
                  clade_comp=['Eukaryota',
                              'Mammalia',
                              'Chordata',
                              'Alveolata',
                              'Viridiplantae',
                              'Fungi'])
plt.close('all')

# %%
#########################################
######### inGmt Reactome ################
#########################################
if args.which == "inGO_path":
    figpath_cur = figpath + "/inGO_path"
    os.makedirs(figpath_cur, exist_ok=True)

    df_pw = pd.read_csv("data/inGO_path.csv")

    dat = df_pw
    dat['ind'] = ind.get_ind_pair([x for x in zip(dat.gene_a, dat.gene_b)])

    sns.set_palette(sns.color_palette("tab10") + sns.color_palette("pastel"))
    targets = sorted(dat.columns.drop(["gene_a", "gene_b", "ind"]).tolist())
    train_and_vis(dat, targets, clfs,  figpath_cur,
                  size_h=6, size_w=6, clf_size_factor=1.7,
                  clade_comp=['Eukaryota',
                              'Mammalia',
                              'Chordata',
                              'Alveolata',
                              'Viridiplantae',
                              'Fungi'])
plt.close('all')

# %%
out.close()
plt.close('all')
