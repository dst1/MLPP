from numba import jit
from bisect import bisect
import numpy as np


class PairsIndex:
    """
    Pairs index caches the indices for each genes and contains several utilities to match a pair with a sorted all pairs index
    """

    def __init__(self, genes, sort=True):
        """
        Initializes a new instance.
        An instance contains the list of genes and a mapping from gene to place in list

        :param genes: list, all genes in index
        :param sort: whether to use a sorted index
        """
        if sort == True:
            self.genes = sorted(genes)
        else:
            self.genes = genes

        self.ind = {k: i for i, k in enumerate(self.genes)}
        self.rev_ind = {i: k for i, k in enumerate(self.genes)}
        self.n = len(genes)
        self.ind_size = (self.n * (self.n - 1)) // 2
        self.i0_inds = [((2 * self.n - i_0 - 1) * i_0) // 2 for i_0 in range(self.n)]

    @classmethod
    def from_file(cls, path, sort=False):
        genes = open(path).read().splitlines()
        return cls(genes, sort=sort)

    def get_ind_solo(self, genes):
        """
        Returns the place on the genes in the list

        Assumes genes are found on the list!!

        :param genes: genes to query
        :return: indexes for these genes
        """
        if type(genes) == str:
            return self.ind[genes]
        else:
            return [self.ind[x] for x in genes]

    def _calc_pair_loc(self, genes):
        """
        Internal function, gets an ordered tuple and calculates its location
        :param genes: list, gene[0]<gene[1], also not the same..
        :return: int, location in pairs list
        """

        i_0, i_1 = self.get_ind_solo(genes)
        ind = self._calc_pair_loc_from_inds(i_0, i_1)
        return ind

    def _calc_pair_loc_from_inds(self, x, y):
        """
        Internal function, gets an ordered tuple and calculates its location
        :param genes: list, gene[0]<gene[1], also not the same..
        :return: int, location in pairs list
        """

        i_0, i_1 = sorted([x, y])
        ind = ((2 * self.n - i_0 - 1) * i_0) // 2 + (i_1 - i_0 - 1)
        return ind

    def get_ind_pair(self, gene_tuples):
        """
        Return the pair index in a sorted, not redundant pairs list (like an upper triangular matrix)

        Assumes genes are found on the list!! and are not the same gene...

        :param gene_tuples: list of tuples, genes are not assumed to be ordered in tuples
        :return: list of indexes
        """

        if type(gene_tuples) == tuple:
            return self._calc_pair_loc(sorted(gene_tuples))
        else:
            return [self._calc_pair_loc(sorted(x)) for x in gene_tuples]

    def get_genes_solo(self, inds):
        """
        Returns the gene in place ind

        Assumes genes are found on the list!!

        :param genes: genes to query
        :return: indexes for these genes
        """
        if type(inds) == int:
            return self.rev_ind[inds]
        else:
            return [self.rev_ind[x] for x in inds]

    def _calc_pair_genes(self, ind, return_inds=False):
        """
        Internal function, gets a location and calulate the indexes of the genes
        returns the genes as a tuple
        :param ind: location in pairs list
        :return: tuple, genes
        """
        i_0 = bisect(self.i0_inds, ind) - 1
        i_1 = ind - self.i0_inds[i_0] + i_0 + 1

        if return_inds:
            return (i_0, i_1)
        return (self.get_genes_solo(i_0), self.get_genes_solo(i_1))

    def get_genes_pair(self, inds):
        """
        Return the pair index in a sorted, not redundant pairs list (like an upper triangular matrix)

        Assumes genes are found on the list!! and are not the same gene...

        :param gene_tuples: list of tuples, genes are not assumed to be ordered in tuples
        :return: list of indexes
        """

        if type(inds) == int:
            return self._calc_pair_genes(inds)
        if type(inds) == range:
            st_0, st_1 = self._calc_pair_genes(inds.start, return_inds=True)

            cur_0 = st_0
            cur_1 = st_1
            out = [(self.rev_ind[cur_0], self.rev_ind[cur_1])]
            for i in range(len(inds) - 1):
                if cur_1 < self.n - 1:
                    cur_1 += 1
                else:
                    cur_0 += 1
                    cur_1 = cur_0 + 1
                out.append((self.rev_ind[cur_0], self.rev_ind[cur_1]))
            return (out)
        else:
            return [self._calc_pair_genes(x) for x in inds]

    def all_other_inds(self, gene_ind):
        inds = [0] * (self.n - 1)
        for i in range(gene_ind):
            inds[i] = ((2 * self.n - i - 1) * i) // 2 + (gene_ind - i - 1)
        for i in range(gene_ind, self.n):
            inds[i - 1] = ((2 * self.n - gene_ind - 1) * gene_ind) // 2 + (i - gene_ind - 1)
        return inds


if __name__ == "__main__":
    ind = {'st': 1013, 'en': 10000}
    params = json.load(open("config/gen_params.json", 'r'))
    pairs = PairsIndex.from_file(params['index_path'])

    dat = pairs.get_genes_pair(range(ind['st'], ind['en'] + 1))
    dat = pairs.all_other_inds(pairs.get_ind_solo("BRCA1"))
    print(len(dat), [pairs.get_genes_pair(int(x)) for x in np.random.choice(dat, 10)])
