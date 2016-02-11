cpdef csr_max_duplicates(
    int n_row,
    int n_col,
    int [:] rptr,
    int[:] rindices,
    double [:] data):
    
    cdef:
        int nnz, row_end, jj, j, i
        double x
    nnz = 0
    row_end = 0
    for i in range(n_row):
        jj = row_end
        row_end = rptr[i+1]
        while( jj < row_end ):
            j = rindices[jj]
            x = data[jj]
            jj += 1
            while( (jj < row_end) and (rindices[jj] == j)):
                if x < data[jj]:
                    x = data[jj]
                jj += 1
            rindices[nnz] = j
            data[nnz] = x
            nnz += 1
        rptr[i+1] = nnz

cpdef transformDayFromLastPurchase(int[:]rptr, int[:] ri, 
                                 double[:] data, double[:] latest_purchased_date,
                                int nrows):
    
    cdef:
        int i, j
        list indices
        double row_max
    for i in xrange(nrows):
        row_max = latest_purchased_date[i]
        for j in range(rptr[i], rptr[i+1]):
            data[j] = row_max - data[j] + 1
