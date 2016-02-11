import numpy as np
from lrec.evaluate.ranking_metric import *
import multiprocessing as mp
import time
import scipy.sparse


def getTestUsers(train, test, cond=(1, float("inf"))):
    low = cond[0]
    high = cond[1]
    test_users = np.ravel(np.where(np.ravel(test.sum(axis=1)) > 0)[0])
    train_user_stat = np.ravel(train.sum(axis=1))
    logical_cond = np.logical_and(
        train_user_stat >= low, train_user_stat <= high)
    train_users = np.ravel(np.where(logical_cond)[0])
    train_users_set = set(train_users.tolist())
    final_test_users = []
    for user in test_users:
        if user in train_users_set:
            final_test_users.append(user)
    return final_test_users


def getTopK(scores,  k):
    topk_items = np.argsort(scores)[::-1]
    topk_items = topk_items.astype(np.int32)
    if k > 0:
        return topk_items[:k]
    else:
        return topk_items


def generateBatches(lst, batch_size):
    batch = []
    for i in range(0, len(lst), batch_size):
        end = min(i + batch_size, len(lst))
        batch.append(lst[i: end])
    return batch


def getUserPurchased(mat, user):
    return mat.indices[range(mat.indptr[user], mat.indptr[user + 1])]


def evalMetrics(train, test, recos, mapk=100, ks=[3, 5, 10, 20]):
    # Input:
    #       train: training data
    #       test: test csr data matrix
    #       recos: recommendation matrix (csr)
    #       mapk : cutoff for map evaluation
    #       ks : @k values for precision and recall
    # Output:
    #         (map@mapk , precision@ks, recall@ks)

    # IMPORTANT: Use this method if you can fit whole test user recommendation
    # in the memory
    testusers = getTestUsers(train, test)

    maps = []
    precs = []
    recalls = []
    for user in testusers:
        if scipy.sparse.issparse(recos):
            reco_score = recos.getrow(user).todense()
        else:
            reco_score = recos[user, :]

        reco_score = np.ravel(reco_score)
#         reco_score = np.ravel(reco_score)
        history = train.getrow(user)
        history_index = history.indices[
            range(history.indptr[0], history.indptr[1])]
        reco_score[history_index] = float("-inf")
        recommended = getTopK(reco_score, mapk)
        user_purchased = getUserPurchased(test, user)

        _apk = apk(user_purchased, recommended, mapk)
        _recalls = []
        _precs = []
        for k in ks:
            _prec = prec(user_purchased, recommended, k)
            _rec = recall(user_purchased, recommended, k)
            _recalls.append(_rec)
            _precs.append(_prec)
        recalls.append(_recalls)
        precs.append(_precs)
        maps.append(_apk)
    return np.mean(maps), np.mean(np.array(precs), axis=0), np.mean(np.array(recalls), axis=0), len(testusers)


def evalMetricsParallelMiniBatch(train_input, train_target, test, model,
                                 mapk=100, ks=[3, 5, 10, 20], batch_size=1000, nprocs=1):
    # Input:
    #       train: training data
    #       test: test csr data matrix
    #       recos: recommendation matrix (csr)
    #       mapk : cutoff for map evaluation
    #       ks : @k values for precision and recall
    # Output:
    #         (map@mapk , precision@ks, recall@ks)

    # IMPORTANT: Use this method if you can fit whole test user recommendation
    # in the memory

    class resultCollector:

        def __init__(self):
            self.results = []
            self.running = 0

        def collect(self, x):
            self.results.append(x)
            self.running -= 1

        def getResult(self):
            maps = 0
            precs = []
            recalls = []
            total_users = 0.0
            for result in self.results:
                _map, _prec, _rec, nuser = result
                maps += _map * nuser
                precs.append(_prec * nuser)
                recalls.append(_rec * nuser)
                total_users += nuser
            _map_score = np.array(maps).sum() / total_users
            prec_score = np.array(precs).sum(axis=0) / total_users
            recall_score = np.array(recalls).sum(axis=0) / total_users

            return _map_score, prec_score, recall_score, int(total_users)

    testusers = getTestUsers(train_target, test)
    testusers_batch = generateBatches(testusers, batch_size)
    collector = resultCollector()
    nprocs = min(nprocs, len(testusers_batch))
    pool = mp.Pool(nprocs)
    for batch_users in testusers_batch:
        reco_score = model.recommend(batch_users, train_input)
        train_batch = train_target[batch_users, :]
        test_batch = test[batch_users, :]
        collector.running += 1
        arg = (train_batch, test_batch, reco_score, mapk, ks)
        pool.apply_async(evalMetrics, args=arg, callback=collector.collect)
        while(collector.running >= nprocs):
            time.sleep(1)
    pool.close()
    pool.join()
    return collector.getResult()


def evalMetricsIterative(train_input, train_target, test, model, mapk=100, ks=[3, 5, 10, 20], cond=None):
    # Input:
    #       train_input: training input for model
    #       train_target : user-item purchase data (csr matrix)
    #       test: test csr data matrix
    #       model
    #       mapk : cutoff for map evaluation
    #       ks : @k values for precision and recall
    # Output:
    #         (map@mapk , precision@ks, recall@ks)

    # IMPORTANT: Use this method to evaluate model on the fly

    testusers = getTestUsers(train_target, test, cond)
    maps = []
    precs = []
    recalls = []
    for user in testusers:
        reco_score = np.ravel(model.recommend(user, train_input))
        history = train_target.getrow(user)
        history_index = history.indices[
            range(history.indptr[0], history.indptr[1])]
        reco_score[history_index] = float("-inf")
        recommended = getTopK(reco_score, mapk)
        user_purchased = getUserPurchased(test, user)

        _apk = apk(user_purchased, recommended, mapk)
        _recalls = []
        _precs = []
        for k in ks:
            _prec = prec(user_purchased, recommended, k)
            _rec = recall(user_purchased, recommended, k)
            _recalls.append(_rec)
            _precs.append(_prec)
        recalls.append(_recalls)
        precs.append(_precs)
        maps.append(_apk)
    return np.mean(maps), np.mean(np.array(precs), axis=0), np.mean(np.array(recalls), axis=0), len(test_users)
