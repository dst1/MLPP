"""
AdaSampling implementation for python, scikit-learn compatible

Author: Doron Stupp

Based on the R package AdaSampling by: Pengyi Yang & Kukulege Dinuka Perera from Sydney U
And the conference papers:
-Yang, P., Ormerod, J., Liu, W., Ma, C., Zomaya, A., Yang, J.(2018)
AdaSampling for positive-unlabeled and label noise learning with bioinformatics applications.
IEEE Transactions on Cybernetics, doi:10.1109/TCYB.2018.2816984

-Yang, P., Liu, W., Yang, J. (2017).
Positive unlabeled learning via wrapper-based adaptive sampling.
Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI), 3273-3279.
"""

from abc import ABCMeta

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble.base import BaseEnsemble
from sklearn.externals.six import with_metaclass
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tqdm import tqdm


class AdaSample(with_metaclass(ABCMeta, BaseEnsemble), ClassifierMixin):
    """
    Basic scikit learn estimator wrapper for using AdaSampling
    Based on the R package AdaSampling and the conference paper:
    AdaSampling for positive-unlabeled and label noise learning with bioinformatics applications.
    Yang, P., Ormerod, J., Liu, W., Ma, C., Zomaya, A., Yang, J.(2018) [doi:10.1109/TCYB.2018.2816984]

    Parameters
    ----------
    clf - base classifier to perform AdaSampling on. Should be a scikit-learn compatible classifier
    with the methods - fit & predict_proba
    """

    def __init__(self, base_estimator, C=10, sampleFactor=1,
                 ensemble_sampleFactor=1, seed=False, n_rounds=5, ensemble_estimator=None):
        super(AdaSample, self).__init__(
            base_estimator=base_estimator)

        self.C = C
        self.sampleFactor = sampleFactor
        self.ensemble_sampleFactor = ensemble_sampleFactor
        self.seed = seed
        self.n_rounds = n_rounds
        self.ensemble_estimator = ensemble_estimator

    def fit(self, X, y, **kargs):
        """
        Fitting function
        Adheres to arguments of the original R package.

        Parameters
        ----------
        X : numpy array, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values - 1 for positive, 0 for negative.
        C : number of ensemble classifiers (default: 1)
        sampleFactor : float, subsampling factor (default: 1)
        ensemble_sampleFactor : float, subsampling factor for fitting the ensemble (default: 1)
        seed : bool, Whether to set seed
               if True, seed is set to iteration index (default: False)
        n_rounds: int, number of AdaSampling rounds (default: 5)
        **kargs: passed into base_extimator fit function
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        self.is_fitted_ = True

        # initialize sampling probablity
        Ps = np.where(y == 1)[0]
        Ns = np.where(y == 0)[0]

        pos_probs = np.ones_like(Ps) / Ps.shape[0]
        una_probas = np.ones_like(Ns) / Ns.shape[0]

        self.adaSamples_ = []
        print("Training AdaSamples..")
        for i in tqdm(range(self.n_rounds)):  # TODO: Maybe till convergence?
            self.adaSamples_.append(
                self._fit_single(self.base_estimator,
                                 X, y,
                                 Ps, Ns,
                                 pos_probs, una_probas,
                                 self.sampleFactor, (i if self.seed else False), **kargs)
            )
            probas = self.adaSamples_[-1].predict_proba(X)
            pos_probs = probas[Ps, 1] / probas[Ps, 1].sum()
            una_probas = probas[Ns, 0] / probas[Ns, 0].sum()

        print("Training {} Classifiers".format(self.C))
        self.estimators_ = []
        if self.ensemble_estimator is not None:
            fit_est=self.ensemble_estimator
        else:
            fit_est=self.base_estimator
        for i in tqdm(range(self.C)):
            self.estimators_.append(
                self._fit_single(fit_est,
                                 X, y,
                                 Ps, Ns,
                                 pos_probs, una_probas,
                                 self.ensemble_sampleFactor,  # fitting using all the data
                                 (i if self.seed else False), **kargs)
            )
        return self

    def _fit_single(self, base_estimator,
                    X, y, Ps, Ns, pos_probs, una_probs, sampleFactor, seed, **kargs):
        """
        Helper function for fitting.
        Similar to singleIter.R

        Parameters
        ----------
        X : numpy array, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values - 1 for positive, 0 for negative.
        Ps : array-like, shape (n_positives,) indeces of positives
        Ns : array-like, shape (n_negatives,) indeces of negatives
        pos_probs : array-like, shape (n_positives,)
                    probability of positives to belong to the positive class
        una_probs : array-like, shape (n_negatives,)
                    probability of unlabeled to belong to the negative class
        sampleFactor : float, subsampling factor (default: 1)
        seed : bool, Whether to set seed
               if True, seed is set to iteration index (default: False)

        Returns
        -------
        clf : fitted classifier object of type self.base_estimator_
        """
        clf = clone(base_estimator)
        if type(seed) == int:
            np.random.seed(seed)

        sampleN = max((Ps.shape[0], Ns.shape[0]))
        ids_p = np.random.choice(Ps, size=int(sampleFactor * sampleN), p=pos_probs)
        X_p = X[ids_p, :]
        y_p = y[ids_p]
        assert (y_p==1).all()

        ids_n = np.random.choice(Ns, size=int(sampleFactor * sampleN), p=una_probs)
        X_n = X[ids_n, :]
        y_n = y[ids_n]
        assert (y_n == 0).all()

        X_train = np.vstack((X_p, X_n))
        y_train = np.concatenate((y_p, y_n))

        clf.fit(X_train, y_train, **kargs)

        return clf

    def predict_proba(self, X, single=False):
        """
        Predicting function. predicts probailities for the ensemble.
        Can be used in AdaSingle mode by setting single=True.
        Then the function predicts based on the first classifier only.

        When single = False, returns the average of predictions for the ensemble.

        Parameters
        ----------
        X : numpy array, shape (n_samples, n_features)
            The training input samples.
        single : bool, whether to predict in mode AdaSingle (default: False)

        Returns
        -------
        y : ndarray, shape (n_samples,2)
            Returns probabilities to belong to the negative class (col 0)
            or the positive class (col 1)
            Similar to other scikit-learn classifiers.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        probas = np.zeros((X.shape[0], 2))
        if single:
            return self.estimators_[0].predict_proba(X)

        for clf in self.estimators_:
            probas += clf.predict_proba(X)

        probas = probas / len(self.estimators_)
        return probas
