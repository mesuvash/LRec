cpdef get_intersection(int[:] purchased, int[:] recommended, int k):
    cdef:
        int i, m
        double hit
        set recos
    recos = set()
    for i in range(k):
        recos.add(recommended[i])
    m = len(purchased)   
    hit = 0.0
    for i in range(m):
        if purchased[i] in recos:
            hit += 1
    return hit 

cpdef double recall(int[:] purchased, int[:] recommended, int k):
    cdef:
        double hit
        int m
    m = len(purchased)
    hit = get_intersection(purchased, recommended, k)
    return hit / m

cpdef double prec(int[:] purchased, int[:] recommended, int k):
    cdef:
        double hit
    hit = get_intersection(purchased, recommended, k)
    return hit / k
        
cpdef double apk(int[:] purchased, int[:] recommended, int k):
    cdef:
        double score
        int i, m, item
        set actual_set
    actual_set = set(purchased)
    score = 0.0
    m = len(purchased)
    for i in range(1, k + 1):
        item = recommended[i-1]
        if  item in actual_set:
            score += prec(purchased, recommended, i)
    return score / min(m, k)
    