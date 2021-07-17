import tables as tb
import pandas as pd
import numpy as np
import threading
from src.tools.PairsIndex import PairsIndex
from src.tools import utils, top_meths


class h5Preds(object):

    def __init__(self, file, index=None, default_path=None):
        if index is None:
            index = h5Preds.load_index()

        assert type(index) == PairsIndex
        self.index = index

        self.default_path = default_path

        self._lock = threading.Lock()
        with self._lock:
            self._h5_file = tb.open_file(file)

        self.default_col = self._h5_file.root[default_path].colnames
        self.default_col = self.default_col[0] if len(
            self.default_col) == 1 else None

    def get_subset(self, path=None, genes=None, ids=None, column=None, return_genes=True):
        """
        Gets the subset of genes / ids from the H5 file.
        If both ids and genes are specified, ids get priority and will be the subset returned.

        :param path: path of data in the H5
        :param genes: list of tuples, gene pairs to use for subset
        :param ids: list of ids, ids for gene pairs (using the same index as used internally)
        :param column: columns to retrieve, predictions / features (default=None, returns all columns)
        :return: df with rows indexed by ids and columns as specified
        """
        assert (genes is not None) or (ids is not None)
        assert (path is not None) or (self.default_path is not None)

        if ids is None:
            ids = self.index.get_ind_pair(genes)

        if path is None:
            path = self.default_path

        if column is not None:
            df = pd.DataFrame({
                x: self._h5_file.root[path].read_coordinates(ids, field=x)
                for x in column})
            df['ind'] = ids
        else:
            df = pd.DataFrame(self._h5_file.root[path].read_coordinates(ids, field=column))
        if return_genes:
            inds_genes = pd.DataFrame(self.index.get_genes_pair(df['ind'].astype('int').tolist()),
                                      columns=['gene_a', 'gene_b'], dtype='category')
            df = pd.concat([df, inds_genes], axis=1)
        return df

    def get_top(self, genes=None, ids=None, column=None, path=None, top_meth=top_meths.top_n, **kargs):
        """[summary]
        Applies top_meth to each gene/id to identify most related genes to the query.
        If both ids and genes are specified, ids get priority and will be the subset returned.

        :param genes: list of strings, defaults to None, genes to use as query either genes or ids are required
        :param ids: list of integers, defaults to None, ids of genes to use as query.
        :param column: column of h5 file to use, defaults to self.default_col if a single column exists else None
        :param path: path of table in file, defaults to self.default_path
        :param top_meth: method used to identify top matches, should recieve a dataframe with ids and target column and returns a list of ids, defaults to top_n
        :param **kargs: passed to top_meth
        :return: {query_gene: [list of matches]} - dict of matching genes/ids (according to query type) to the query.
        """
        assert (genes is not None) or (ids is not None)
        assert (path is not None) or (self.default_path is not None)

        if column is None:
            column = self.default_col
        assert column is not None

        if ids is None:
            use_genes = True
            ids = self.index.get_ind_solo(genes)
        if path is None:
            path = self.default_path

        top = {}
        for i, id_ in enumerate(ids):
            others = self.index.all_other_inds(id_)
            df = self.get_subset(path=path, ids=others, column=['ind', column])

            k = genes[i] if use_genes else id_

            tmp = top_meth(df, **kargs)

            top[k] = df.loc[df['ind'].isin(tmp), :].sort_values(by=column, ascending=False)
            top[k] = top[k]
            # top[k] = [x[0] if x[1] == id_ else x[1] for x in [
            #     self.index._calc_pair_genes(y, return_inds=True) for y in top[k]]]
            #
            # if use_genes:
            #     top[k] = [self.index.get_genes_solo(x) for x in top[k]]
        return top

    def read_all(self, path=None, field=None):
        """
        reads the entire HDF5 file to memory and parses inds back to genes.
        Can be VERY memory expensive...

        :return: dataframe
        """
        if path is None:
            path = self.default_path

        df = pd.DataFrame(self._h5_file.root[path].read(field=field))
        return df

    # TODO: calc explanations
    # def get_explanations(self, path, genes=None, ids=None, columns=None):
    #    assert (genes is not None) or (ids is not None)
    #    pass

    def to_numpy(self, field, hdf_path=None, file_path=None):
        """
        Converts a column in the hdf to numpy array and (optionally) saves to file
        :param field: column in the hdf file to convert
        :param hdf_path: path in the hdf file to the table, defaults to self.default_path
        :param file_path: file path to save numpy array to, default=None - doesn't save
        :return: numpy array (nXn) where n=# of genes
        """
        if hdf_path is None:
            hdf_path = self.default_path
        x = self._h5_file.root[hdf_path].read(field=field)
        n = self.index.n
        adj = np.zeros((n, n), dtype=np.float32)

        adj[np.triu_indices(n, 1)] = x

        adj = adj + adj.T
        np.fill_diagonal(adj, -np.inf)

        if file_path is not None:
            h5 = tb.open_file(file_path + ".hdf", 'w')
            root = h5.root
            h5.create_array(root, "adjacency", adj)
            h5.close()
        return adj

    @staticmethod
    def load_index(ind_path="data/processed/NPP/genes_list.txt"):
        pairs = PairsIndex.from_file(ind_path)
        return pairs

    def close(self):
        with self._lock:
            return self._h5_file.close()


if __name__ == "__main__":
    test = 1
    if test == 1:
        AnyLink = h5Preds(
            "data/processed/Predictions/AnyLink_DEFAULT.hdf5", default_path="/data/AnyLink")
        print(AnyLink)
        gene = "ARSA"
        AnyLink.get_top(genes=[gene], column="AnyLink", **{'n': 50})
        # b.remove((gene, gene))
        # inds = AnyLink.index.get_ind_pair(b)
        # df = AnyLink.get_subset(ids=inds)

        # a = AnyLink.index.get_genes_pair(df.ind)
        s = gene + "\n"
        for x in a:
            if x[0] == gene:
                s += x[1] + "\n"
            else:
                s += x[0] + "\n"
        print(s)
        df = AnyLink.read_all()
        print(df)
    if test == 2:
        PP_comp = h5Preds("data/processed/Predictions/PP_comp.h5",
                          default_path="/data/PP_comp")
        print(PP_comp)
