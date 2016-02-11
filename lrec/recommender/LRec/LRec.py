from sklearn.linear_model import LogisticRegression, Ridge
import scipy.sparse
import numpy as np
from lrec.recommender.LRec.base import BaseLinear
from lrec.parallel.ipythonParallelLinear import ParallelRunner


class LRec(BaseLinear):

    def __init__(self, arg):
        super(LRec, self).__init__()
        self.arg = arg
        self.target = None
        self.__initargs()

    def __initargs(self):
        self.l2 = self.arg.l2
        self.loss = self.arg.loss

    def __getLearner(self):
        if self.arg.loss == "squared":
            return Ridge(alpha=self.l2, fit_intercept=False)
        elif self.arg.loss == "logistic":
            return LogisticRegression(C=self.l2, fit_intercept=False)
        else:
            raise NotImplementedError(
                "Model %s not implemented" % (self.arg.loss))

    def fit(self, train_input, target_indices=None):
        import numpy as np
        import scipy.sparse
        models = []
        train_target = train_input.T
        train_input = train_input.T

        if target_indices is not None:
            train_target = train_target[:, target_indices]
        else:
            target_indices = range(train_target.shape[1])
        # for fast column access
        train_target = train_target.tocsc()
        for i, index in enumerate(target_indices):
            learner = self.__getLearner()
            y = np.ravel(train_target.getcol(i).todense())
            learner.fit(train_input, y)
            models.append(learner.coef_)
        self.sim = np.vstack(models).T
        return target_indices, self.sim

    def fit_parallel(self, train_input,
                     target_indices=None, num_procs=4,
                     batch_size=1000):
        prunner = ParallelRunner(self, num_procs, batch_size)
        indices, sim = prunner.fit(train_input)
        self.sim = sim
        return indices, self.sim

    def recommend_all(self, train_input):
        score = np.dot(self.sim.T * train_input)
        return score

    def recommend(self, users, train_input):
        reco = (self.sim[:, users].T * train_input)
        if scipy.sparse.issparse(reco):
            return reco.todense()
        else:
            return reco
