import scipy.sparse
import numpy as np
import envoy
import progressbar
import sys
from lrec.utils.data_utils.data_helpers import coo_tocsr


class Data(object):

    def __init__(self):
        self.users = {}
        self.items = {}
        self.nusers = 0
        self.nitems = 0
        self.include_time = False

    def update_user_item(self, user, item):
        if user not in self.users:
            self.users[user] = self.nusers
            self.nusers += 1
        if item not in self.items:
            self.items[item] = self.nitems
            self.nitems += 1

    def import_data(self, filename, parser, shape=None, num_headers=0, debug=False):
        r = envoy.run('wc -l {}'.format(filename))
        num_lines = int(r.std_out.strip().partition(' ')[0])
        bar = progressbar.ProgressBar(maxval=num_lines, widgets=["Loading data: ",
                                                                 progressbar.Bar(
                                                                     '=', '[', ']'),
                                                                 ' ', progressbar.Percentage(),

                                                                 ' ', progressbar.ETA()]).start()
        I, J, V = [], [], []
        with open(filename) as f:
            for i in range(num_headers):
                f.readline()
            for i, line in enumerate(f):
                if (i % 1000) == 0:
                    bar.update(i % bar.maxval)
                try:
                    userid, itemid, rating = parser.parse(line)
                    self.update_user_item(userid, itemid)
                    uid = self.users[userid]
                    iid = self.items[itemid]
                    I.append(uid)
                    J.append(iid)
                    V.append(float(rating))
                except:
                    if debug:
                        print "Ignoring Input: ", line,
                    continue
        bar.finish()
        if shape is not None:
            _shape = (self.nusers if shape[0] is None else shape[0],
                      self.nitems if shape[1] is None else shape[1])
            R = scipy.sparse.coo_matrix(
                (V, (I, J)), shape=_shape)
        else:
            R = scipy.sparse.coo_matrix(
                (V, (I, J)), shape=(self.nusers, self.nitems))
        self.R = coo_tocsr(R)
        sys.stdout.flush()
        return self.R

    def filter(self, n_users=5, n_items=5, iscount=False):
        while True:
            if iscount:
                Rcp = self.R.copy()
                Rcp.data[:] = 1.0
            else:
                Rcp = self.R
            user_stats = Rcp.sum(axis=1)
            item_stats = Rcp.sum(axis=0)
            filter_user = np.ravel((user_stats < n_users) * 1)
            filter_user_cum = np.cumsum(filter_user)
            filter_item = np.ravel((item_stats < n_items) * 1)
            filter_item_cum = np.cumsum(filter_item)
            if (filter_user_cum[-1] == 0) and (filter_item_cum[-1] == 0):
                break

            m, n = self.R.shape

            # filter User item
            I, J, V = [], [], []
            data, ri, rptr = self.R.data, self.R.indices, self.R.indptr
            for i in xrange(m):
                indices = range(rptr[i], rptr[i + 1])
                items = ri[indices]
                ratings = data[indices]
                for j, item in enumerate(items):
                    if (filter_user[i] == 0) and (filter_item[item] == 0):
                        I.append(i - filter_user_cum[i])
                        J.append(item - filter_item_cum[item])
                        V.append(ratings[j])
            R = scipy.sparse.coo_matrix((V, (I, J)),
                                        shape=(m - filter_user_cum[-1],
                                               n - filter_item_cum[-1]))
            self.R = R.tocsr()
            # self.R = coo_tocsr(R)

            inv_users = {v: k for k, v in self.users.items()}
            inv_items = {v: k for k, v in self.items.items()}

            for i in range(m):
                if filter_user[i] == 1:
                    del self.users[inv_users[i]]
                else:
                    self.users[inv_users[i]] -= filter_user_cum[i]

            for i in range(n):
                if filter_item[i] == 1:
                    del self.items[inv_items[i]]
                else:
                    self.items[inv_items[i]] -= filter_item_cum[i]


def loadDataset(filename, usermap, itemmap, parser, shape=None):
    r = envoy.run('wc -l {}'.format(filename))
    num_lines = int(r.std_out.strip().partition(' ')[0])
    bar = progressbar.ProgressBar(maxval=num_lines, widgets=["Loading data: ",
                                                             progressbar.Bar(
                                                                 '=', '[', ']'),
                                                             ' ', progressbar.Percentage(),

                                                             ' ', progressbar.ETA()]).start()
    I, J, V = [], [], []
    cold = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if (i % 1000) == 0:
                bar.update(i % bar.maxval)
            userid, itemid, rating = parser.parse(line)
            if userid not in usermap or itemid not in itemmap:
                cold.append((userid, itemid, rating))
                continue
            uid = usermap[userid]
            iid = itemmap[itemid]
            I.append(uid)
            J.append(iid)
            V.append(float(rating))
    bar.finish()
    if shape is not None:
        R = scipy.sparse.coo_matrix((V, (I, J)), shape=shape)
    else:
        R = scipy.sparse.coo_matrix(
            (V, (I, J)), shape=(len(usermap), len(itemmap)))
    R = coo_tocsr(R)

    return R, cold


def loadSideInfo(filename, targetmap, parser, shape=None):
    r = envoy.run('wc -l {}'.format(filename))
    num_lines = int(r.std_out.strip().partition(' ')[0])
    bar = progressbar.ProgressBar(maxval=num_lines, widgets=["Loading data: ",
                                                             progressbar.Bar(
                                                                 '=', '[', ']'),
                                                             ' ', progressbar.Percentage(),

                                                             ' ', progressbar.ETA()]).start()
    I, J, V = [], [], []
    cold = []
    counter = 0
    feature_map = {}
    with open(filename) as f:
        for i, line in enumerate(f):
            if (i % 1000) == 0:
                bar.update(i % bar.maxval)
            keyid, featureid = parser.parse(line)
            if keyid not in targetmap:
                continue
            if featureid not in feature_map:
                feature_map[featureid] = counter
                counter += 1
            kid = targetmap[keyid]
            fid = feature_map[featureid]
            I.append(kid)
            J.append(fid)
            V.append(1.0)
    bar.finish()
    if shape is not None:
        R = scipy.sparse.coo_matrix((V, (I, J)), shape=shape)
    else:
        R = scipy.sparse.coo_matrix(
            (V, (I, J)), shape=(len(targetmap), len(feature_map)))
    R = coo_tocsr(R)

    return R, feature_map
