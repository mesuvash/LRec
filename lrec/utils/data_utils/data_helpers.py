from scipy.sparse._sparsetools import coo_tocsr as scipy_coo_tocsr
from scipy.sparse.sputils import get_index_dtype, upcast
import scipy.sparse
from lrec.utils.data_utils.data_cython_helpers import csr_max_duplicates
import numpy as np

def coo_tocsr(coo_mat):
    M, N = coo_mat.shape
    idx_dtype = get_index_dtype((coo_mat.row, coo_mat.col),
                                maxval=max(coo_mat.nnz, N))
    indptr = np.empty(M + 1, dtype=idx_dtype)
    indices = np.empty(coo_mat.nnz, dtype=idx_dtype)
    data = np.empty(coo_mat.nnz, dtype=upcast(coo_mat.dtype))

    scipy_coo_tocsr(M, N, coo_mat.nnz,
              coo_mat.row.astype(idx_dtype),
              coo_mat.col.astype(idx_dtype),
              coo_mat.data,
              indptr, indices, data)
    A = scipy.sparse.csr_matrix((data, indices, indptr), shape=coo_mat.shape)
    A.sort_indices()
    csr_max_duplicates(M, N, A.indptr, A.indices, A.data)
    A.prune()
    A.has_canonical_format = True
    return A


def transformDayFromLastPurchase(X):
    offset = 1
    latest_purchased_date = np.ravel(X.max(axis=1).todense())
    data, rptr, ri = X.data, X.indptr, X.indices
    result = np.array([])
    for i in xrange(X.shape[0]):
        indices = range(rptr[i], rptr[i + 1])
        result = np.append(result, latest_purchased_date[
                           i] - data[indices] + offset)
    X.data = result
