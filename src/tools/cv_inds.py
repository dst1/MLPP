import datetime
import json
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd
import tables as tb

from src.tools import utils
from src.tools.PairsIndex import PairsIndex


tempfile.tempdir = "data/tmp/"
if not os.path.exists(tempfile.tempdir):
    os.makedirs(tempfile.tempdir, exist_ok=True)

PM_SPLITS=['train','test','C1','C2','C3']

class CV(object):
    """
    CV class, implements methods to get CV folds based on pre-defined features and target
    """

    def __init__(self, dat, target, ind,
                 CV_path=None, splits_save_dir=None,
                 feats=[], 
                 params=None, gen_folds=True):
        """
        Initializes a CV splits object
        :param dat: DataFrame, indexes and target value
        :param target: name of target column
        :param ind: PairsIndex object
        :param CV_path: path to save splits to
        :param feats: name of feature types to load
        :param params: json config file with general parameters for CV creation
        """

        self.genes = np.unique([ind._calc_pair_genes(i, return_inds=True) for i in dat['ind']])
        self.pos = dat.loc[dat[target] == 1, "ind"]
        self.target = target
        self.ind = ind
        self.splits_save_dir = splits_save_dir

        self.pos_genes = [ind._calc_pair_genes(i, return_inds=True) for i in self.pos]
        self.pos_genes = np.unique(self.pos_genes)

        if type(params) == dict:
            self.params = params
        elif type(params) == str:
            self.params = json.load(open(params, 'r'))
        else:
            raise TypeError("Params should be either a dictionary of parameters or a path to a json file")

        self.feats = []

        if gen_folds:
            self.gen_folds()

            if CV_path is not None:
                self._save_splits(CV_path)

    def gen_folds(self):
        self.cv_folds = []
        for i in range(self.params['n_splits']):
            print(f"CV {i + 1} / {self.params['n_splits']}:")
            print("Generating negatives")
            neg, neg_genes = self._generate_neg()

            print("Train / Test split")
            train_X, train_y, test_X, test_y, test_genes = self._stratify(neg, neg_genes)

            train_genes = np.union1d(neg_genes, self.pos_genes)
            train_genes = train_genes[~np.isin(train_genes, test_genes)]

            print("Calculate Park Marcotte splits for test set")
            Cs = self._calculate_PM(test_X, train_genes)
            print("Sizes of Park Marcotte splits: " +
                  ", ".join([f"C{i}: {sum(Cs == i)} ({sum(Cs[np.where(test_y == 1)[0]] == i) / sum(Cs == i):.1%} pos)"
                             for i in range(1, 4)]))
            self.cv_folds.append({
                "train": (train_X, train_y),
                "test": (test_X, test_y)})
            for j in range(1, 4):
                self.cv_folds[-1][f"C{j}"] = (test_X[np.where(Cs == j)[0]],
                                              test_y[np.where(Cs == j)[0]])

            if self.splits_save_dir is not None:
                os.makedirs(self.splits_save_dir, exist_ok=True)
                self.cv_folds[-1] = single_CV_fold(self.cv_folds[-1],
                                                   feats=self.feats,
                                                   params=self.params,
                                                   ind=self.ind,
                                                   save_dir=self.splits_save_dir + f"/CV{i}_")
            else:
                self.cv_folds[-1] = single_CV_fold(self.cv_folds[-1],
                                                   params=self.params,
                                                   ind=self.ind,
                                                   feats=self.feats)

    def _generate_neg(self):
        """
        Generates negative indexes
        """
        n = self.pos.shape[0]
        n0 = int(n / self.params['pos_ratio'])

        # sample (neg) genes
        if self.params['pos_only']:
            neg_genes = self.pos_genes
        else:
            neg_genes = np.random.choice(np.arange(0, self.ind.n),
                                         min(int(self.pos_genes.shape[0] * 1.2 * np.sqrt(1 / self.params['pos_ratio'])),
                                             self.ind.n),
                                         replace=False)

        # sample neg:

        neg_samp_size = int(2 * n0 / self.params['pos_ratio'])
        neg = np.zeros((neg_samp_size, 2), dtype=np.int64)
        neg[:, 0] = np.random.choice(neg_genes, neg_samp_size)
        neg[:, 1] = np.random.choice(neg_genes, neg_samp_size)
        neg = neg[np.where(neg[:, 0] != neg[:, 1])[0], :]
        neg.sort(axis=1)

        # neg = [tuple(np.random.choice(neg_genes, 2, replace=False)) for _ in range(int(n0/self.params['pos_ratio'] * 2))]
        neg = [self.ind._calc_pair_loc_from_inds(*x) for x in neg.tolist()]

        # check duplicates:
        neg = np.unique(neg)

        # check pos in neg:
        neg = neg[~np.isin(neg, self.pos)]

        # sample negatives to right size:
        assert neg.shape[0] >= n0
        neg = np.random.choice(neg, n0, replace=False)
        neg_genes = [self.ind._calc_pair_genes(i, return_inds=True) for i in neg]
        neg_genes = np.unique(neg_genes)
        return neg, neg_genes

    def _paralog_stratified_test_genes(self, neg_genes):
        """
        Iteratively adds genes as test genes
        for each iteration a small proportion is sampled and all genes similar to the sample (by bitscore) are also added
        :param neg_genes: negative genes sampled by _generate_neg
        :return: test_genes a dict with a list of (inds) of test genes for positive and negative
        """
        h5 = tb.open_file(self.params['bitscore_mat'], "r")
        bitscore_mat = h5.root["/data"]#.read()
        #bitscore_mat = np.nan_to_num(bitscore_mat)

        unique_genes = {'pos': self.pos_genes,
                        'neg': neg_genes}
        test_genes = {x: np.array([]) for x in unique_genes}

        final_size = {x: (len(unique_genes[x]) * self.params['prop_test']) for x in test_genes}
        while any([len(test_genes[x]) < final_size[x] for x in test_genes]):
            for x in test_genes:
                if len(test_genes[x]) >= final_size[x]:
                    continue
                tmp = np.random.choice(unique_genes[x][~np.isin(unique_genes[x], test_genes[x])],
                                       int((final_size[x] - len(test_genes[x])) // 3) + 1)
                inds_add = tmp[:]
                for i in tmp:
                    tmp_bs = np.nan_to_num(bitscore_mat[i,:])
                    inds_add = np.append(inds_add,
                                         unique_genes[x][np.where(tmp_bs[unique_genes[x]] > self.params["bitscore_threshold"])[0]])
                test_genes[x] = np.append(test_genes[x], np.unique(inds_add))
        h5.close()
        del bitscore_mat
        return test_genes

    def _stratify(self, neg, neg_genes):
        """
        Stratifies test genes for PM splits and generates train,test splits accordingly
        """
        # sizes to stratify by
        n = self.pos.shape[0] + neg.shape[0]
        train_size = self.params['train_size']
        test_size = 1 - train_size

        pos_rat = self.pos.shape[0] / n
        # find unique genes and allocate prop_test_genes to test
        test_genes = self._paralog_stratified_test_genes(neg_genes)
        print(", ".join(["Test genes {} size: {}".format(k, len(v)) for k, v in test_genes.items()]))

        test_genes = np.unique(np.concatenate((test_genes['pos'], test_genes['neg'])))

        ##### GEN dat #######
        target = "target"
        dat_pos = pd.DataFrame.from_records([self.ind._calc_pair_genes(x, return_inds=True) for x in self.pos],
                                            columns=["gene_a", "gene_b"])
        dat_pos[target] = 1
        dat_neg = pd.DataFrame.from_records([self.ind._calc_pair_genes(x, return_inds=True) for x in neg],
                                            columns=["gene_a", "gene_b"])
        dat_neg[target] = 0

        dat = pd.concat([dat_pos, dat_neg],
                        ignore_index=True, sort=False, copy=True)
        del dat_pos, dat_neg

        ##### calculate for TEST ######
        ps_test = np.zeros(n)

        # get test genes (p=1)
        ps_test[np.where(dat["gene_a"].isin(test_genes) | dat["gene_b"].isin(test_genes))] = 1

        # calculate test neg C2,3
        ps_test_neg = ps_test.copy()
        ps_test_neg[np.where(dat[target] == 1)] = 0
        test_neg_inds = np.random.choice(n, np.sum(ps_test_neg > 0), p=ps_test_neg / ps_test_neg.sum(), replace=False)

        # calculate test pos C2,3
        ps_test_pos = ps_test.copy()
        ps_test_pos[np.where(dat[target] == 0)] = 0
        test_pos_inds = np.random.choice(n, np.sum(ps_test_pos > 0), p=ps_test_pos / ps_test_pos.sum(), replace=False)

        # calculate test neg C1
        ps_test_neg = 1 - ps_test.copy()
        ps_test_neg[np.where(dat[target] == 1)] = 0
        test_neg_inds = np.append(test_neg_inds,
                                  np.random.choice(n, max(
                                      int(n * (1 - pos_rat) * test_size - test_neg_inds.size),
                                      int(n * (1 - pos_rat) * test_size * 0.2)),
                                                   p=ps_test_neg / ps_test_neg.sum(),
                                                   replace=False))

        # calculate test pos C1
        ps_test_pos = 1 - ps_test.copy()
        ps_test_pos[np.where(dat[target] == 0)] = 0
        test_pos_inds = np.append(test_pos_inds,
                                  np.random.choice(n, max(
                                      int(n * (pos_rat) * test_size - test_pos_inds.size),
                                      int(n * (pos_rat) * test_size * 0.2)),
                                                   p=ps_test_pos / ps_test_pos.sum(),
                                                   replace=False))

        print(f"test size: {len(test_pos_inds) + len(test_neg_inds)} " +
              f"({(len(test_pos_inds) + len(test_neg_inds)) / n:.3f})," +
              f"pos: {len(test_pos_inds)} ({len(test_pos_inds) / n:.3f})," +
              f"neg: {len(test_neg_inds)} ({len(test_neg_inds) / n:.3f})")

        ###### calculate for TRAIN #######
        ps_train = np.ones(n)

        # get rid of test genes (p=0)
        ps_train[np.where(dat['gene_a'].isin(test_genes) | dat['gene_b'].isin(test_genes))] = 0

        # calculate train neg
        ps_train_neg = ps_train.copy()
        ps_train_neg[np.where(dat[target] == 1)] = 0
        ps_train_neg[test_neg_inds] = 0  # Prevent leakage
        train_neg_inds = np.random.choice(n, min(int(n * (1 - pos_rat) * train_size),
                                                 sum(ps_train_neg > 0)),
                                          p=ps_train_neg / ps_train_neg.sum(), replace=False)

        # calculate train pos
        ps_train_pos = ps_train.copy()
        ps_train_pos[np.where(dat[target] == 0)] = 0
        ps_train_pos[test_pos_inds] = 0  # Prevent leakage
        train_pos_inds = np.random.choice(n, min(int(n * pos_rat * train_size), sum(ps_train_pos > 0)),
                                          p=ps_train_pos / ps_train_pos.sum(), replace=False)

        ###### MERGE ########
        train = np.concatenate((train_pos_inds, train_neg_inds))
        train = dat.iloc[train, :]
        train = train.sample(frac=1)

        train_inds = np.array([self.ind._calc_pair_loc_from_inds(x, y)
                               for x, y in zip(train['gene_a'], train['gene_b'])])
        train_y = train[target].values
        # train_inds, train_y = shuffle(train_inds, train_y)

        test = np.concatenate((test_pos_inds, test_neg_inds))
        test = dat.iloc[test, :]
        test = test.sample(frac=1)

        test_inds = np.array([self.ind._calc_pair_loc_from_inds(x, y)
                              for x, y in zip(test['gene_a'], test['gene_b'])])
        test_y = test[target].values

        return train_inds, train_y, test_inds, test_y, test_genes

    def _calculate_PM(self, test_inds, train_genes):
        # get test genes
        test = pd.DataFrame.from_records([self.ind._calc_pair_genes(x, return_inds=True) for x in test_inds],
                                         columns=["gene_a", "gene_b"])

        # calculate Park&Marcotte allocation per index
        Cs = 3 - (test.gene_a.isin(train_genes).astype(int) + test.gene_b.isin(train_genes).astype(int))
        return Cs

    def _save_splits(self, save_path):
        name = re.sub("[()]", "", self.target)
        path_all = save_path

        os.system("mkdir -p {}".format(path_all))
        for i, fold in enumerate(self.cv_folds):
            for k, v in fold.inds_dict.items():
                df = pd.DataFrame({"ind": v[0], self.target: v[1]})
                df.to_csv(path_all + f"/{name}_CV{i}_{k}.csv", index=False)


def from_dir(dat, target, ind,
             CV_path=None, splits_save_dir=None,
             feats=[], 
             params=None):
    """
    Assumes dat, target, params, and ind are constructed the same.

    :param CV_path: a folder containing the cvs
    :param splits_save_dir: a folder containing hdfs
    :return:
    """
    obj = CV(dat, target, ind, CV_path, splits_save_dir, feats, params, gen_folds=False)
    obj.cv_folds = []

    name = re.sub("[()]", "", obj.target)

    for i in range(obj.params['n_splits']):
        obj.cv_folds.append({})
        for k in PM_SPLITS:
            df = pd.read_csv(CV_path + f"/{name}_CV{i}_{k}.csv")
            obj.cv_folds[-1][k] = (df['ind'].values, df[target].values)
        obj.cv_folds[i] = single_CV_fold(obj.cv_folds[i],
                                           feats=feats,
                                           params=params,
                                           ind=obj.ind,
                                           save_dir=splits_save_dir + f"/CV{i}_")
    return obj
class single_CV_fold(object):
    def __init__(self, inds_dict, feats, params,ind, save_dir=None):
        self.inds_dict = inds_dict
        self.feats = feats
        self.params=params
        self.ind=ind
        self.func = _get_func(params["func"])
        if save_dir is not None:
            self.save_dir = save_dir
        else:
            self.save_dir_tmp = tempfile.TemporaryDirectory()
            self.save_dir = self.save_dir_tmp.name+"/"

        if len(self.feats)==0:
            self.NPP, _ = utils.load_NPP_mat(path=params['npp_path'], scale_mat=params['scale'])
            self.clade_dict, _ = utils.load_clades(size=params['min_clade_size'])

    def __getitem__(self, key):
        if not (key in self.inds_dict.keys()):
            raise KeyError(f"{key} is not a valid split - {', '.join(self.inds_dict.keys())}")
        else:
            file_path = self.save_dir + f"{key}.hdf"
            if os.path.exists(file_path):
                print(f"Reading from h5..", file=sys.stderr)
                df = pd.read_hdf(file_path, 'df')
            else:
                df = pd.DataFrame({"ind": self.inds_dict[key][0], "target": self.inds_dict[key][1]}, dtype="uint32")
                if len(self.feats)==0:
                    df = get_cor_clades(self.NPP, self.inds_dict[key][0], self.ind,
                                           self.clade_dict, self.func)
                    df['target'] = self.inds_dict[key][1]
                else:
                    for feature, h5_tab in self.feats.items():
                        print(f"Loading {feature}..", file=sys.stderr)
                        st = datetime.datetime.now()
                        df_feat = pd.DataFrame(h5_tab.read_coordinates(list(self.inds_dict[key][0])))
                        df_feat = df_feat.add_suffix("_" + feature)

                        df = pd.merge(df, df_feat, left_on="ind", right_on="ind_" + feature, suffixes=("", "_" + feature))
                        df.drop(columns=["ind_" + feature], inplace=True)
                        print(f"Loaded {feature}! elapsed: {(datetime.datetime.now() - st).total_seconds():.2f}s",
                              file=sys.stderr)
                print(f"Writing to h5..", file=sys.stderr)
                st = datetime.datetime.now()
                df.to_hdf(file_path, 'df', complevel=1, complib='blosc')
                print(f"Wrote to file! elapsed: {(datetime.datetime.now() - st).total_seconds():.2f}s",
                      file=sys.stderr)
            return df

    def filter_paralogous_pairs(self, df_para, df=None, key=None):
        assert (df is not None) or (key is not None), "Either key or df should be specified"
        if df is None:
            df = self.__getitem__(key)

        print("Filtering paralogs", file=sys.stderr)
        df = df.loc[~df['ind'].isin(df_para['ind']), :]
        return df

def _get_func(s):
    s=s.lower()
    if s=="cor":
        return utils.cor
    if s=="cov":
        return utils.cov
    if s=="hamming":
        return utils.manhattan
    if s=="jaccard":
        return utils.jaccard
    return utils.cor

def get_Xy(dat):
    """
    Splits a structured df (generated by a single_CV_fold object) to X,y
    """
    X_s = dat.drop(columns=["ind", "target"])
    feats = X_s.columns.tolist()

    X_s = X_s.values
    y_s = dat['target'].values
    return X_s, y_s, feats

def get_cor_clades(NPP, inds, ind, clade_dict, cor_func=utils.cov):
    '''
    Calculates the correlation between pairs across different clades
    :param NPP: the NPP matrix
    :param inds: indeces to calc on
    :param ind: PairsIndex object
    :param clade_dict: dict of lists, each list contains the column numbers for the clade
    :return: a data frame
    '''

    df = pd.DataFrame({'ind':inds})
    ind_a, ind_b = zip(*[ind._calc_pair_genes(x,return_inds=True) for x in inds])
    ind_a, ind_b = np.array(ind_a), np.array(ind_b)
    for clade in clade_dict:
        df[clade] = utils.calc_any(NPP[:,clade_dict[clade]], ind_a, ind_b, cor_func)
    return df

if __name__ == "__main__":
    params = json.load(open("config/gen_params.json", 'r'))
    params['val_size'] = 0.005
    params['n_splits'] = 1

    # %% LOAD
    NPP, NPP_genes = utils.load_NPP_mat(path=params['npp_path'], scale_mat=params['scale'])
    ind = PairsIndex(NPP_genes)

    df_pw = pd.read_csv(
        "data/processed/inReactome_path.csv")
    # df_pw = df_pw.sample(frac=0.1)
    df_pw['ind'] = ind.get_ind_pair([x for x in zip(df_pw.gene_a, df_pw.gene_b)])

    cv_folds = CV(df_pw, "Cell_Cycle", ind, feats=["clade_cov"], params=params)
    tmp = cv_folds.cv_folds[0]['C3']
