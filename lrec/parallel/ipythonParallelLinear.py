import numpy as np
import scipy.sparse
from copy import deepcopy
import time
import multiprocessing as mp


def generateBatches(lst, batch_size):
    batch = []
    for i in range(0, len(lst), batch_size):
        end = min(i + batch_size, len(lst))
        batch.append(lst[i: end])
    return batch


def parallelRunnerHelper(model, train_input, batch):
    import numpy as np
    import scipy.sparse
    return model.fit(train_input, batch)


def argsort(seq):
    return [x for x, y in sorted(enumerate(seq), key=lambda x: x[1])]


class ResultCollector:

    def __init__(self):
        self.results = []
        self.running = 0

    def collect(self, x):
        self.results.append(x)
        self.running -= 1

    def getResult(self):
        indices = []
        sims = []
        sorted_indices = argsort(map(lambda x: x[0][0],  self.results))
        for index in sorted_indices:
            _indices, _sims = self.results[index]
            indices.extend(_indices)
            sims.append(_sims)
        return indices, np.hstack(sims).T
        # scipy.sparse.hstack(sims, format="csr").T


class ParallelRunner(object):

    def __init__(self, model, nprocs=5, batch_size=1000):
        super(ParallelRunner, self).__init__()
        self.model = model
        self.nprocs = nprocs
        self.batch_size = batch_size

    def fit(self, train_input, target_indices=None):
        nprocs = self.nprocs
        if target_indices is None:
            num = train_input.shape[0]
            batch_indices = generateBatches(range(num), self.batch_size)
        else:
            batch_indices = generateBatches(target_indices, self.batch_size)
        print "Model Learning Started"
        collector = ResultCollector()
        nprocs = min(nprocs, len(batch_indices))
        pool = mp.Pool(nprocs)
        for batch_users in batch_indices:
            args = (deepcopy(self.model), train_input,
                    batch_users)
            pool.apply_async(parallelRunnerHelper, args=args,
                             callback=collector.collect)
            while(collector.running >= nprocs):
                time.sleep(1)
        pool.close()
        pool.join()
        indices, sim = collector.getResult()
        print "Model Learning Ended"
        self.model.sim = sim
        return indices, sim
