import numpy as np

def lu(A):
    # TODO
    U = None
    L = None
    m = A.shape[0]
    U = A.astype(float)
    L = np.identity(m)
    for k in range (m-1):
        for j in range (k+1,m):
            L[j][k] = np.divide(U[j][k],U[k][k])
            U[j][k:m] = U[j][k:m] - np.multiply(L[j][k],U[k][k:m])
    return (L, U)

def maxabs_idx(A):
    # TODO
    i, j = (0, 0)
    max=np.absolute(A[0][0])
    (m,n) = A.shape
    for k in range (m):
        for l in range (n):
             if np.absolute(A[k][l])>max:
                (i,j)=(k,l)
                max=np.absolute(A[k][l])
    return (i, j)

def lu_complete(A):
    # TODO
    U = None
    L = None
    P = None
    Q = None
    m = A.shape[0]
    A = A.astype(float)
    U = A.copy()
    L = np.identity(A.shape[0], dtype=A.dtype)
    P = np.identity(A.shape[0], dtype=A.dtype)
    Q = np.identity(A.shape[0], dtype=A.dtype)

    for k in range(m-1):
        [r, c] = np.array(maxabs_idx(U[k:m, k:m]))#taking the index of the max value of whole matrix
        U[[k, r+k], k:m] = U[[r+k, k], k:m]#reduced matrix interchange
        U[:, [k, c+k]] = U[:, [c+k, k]]
        L[[k, r+k], 0:k] = L[[r+k, k], 0:k]
        Q[:, [k, c+k]] = Q[:, [c+k, k]]#column change in permutation matrix Qk
        P[[k, r+k], :] = P[[r+k, k], :]#row change in permutation matrix Pk
        for j in range(k+1, m):
            L[j, k] = np.divide(U[j, k], U[k, k])
            U[j, k:] = np.subtract(U[j, k:], np.multiply(L[j, k],U[k, k:]))
    return (P, Q, L, U)