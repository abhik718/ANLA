import numpy as np

def sign(x):
    if (x == 0):
        s = 1
        return (s)
    else:
        s = x / (np.linalg.norm(x))
        return (s)

def implicit_qr(A):
    W = None
    R = None
    # TODO
    R = A.astype(complex)
    (m,n) = R.shape
    W = np.zeros((m,n)).astype(complex)
    for k in range(n):
        x = R[k:m,k].copy()
        e1 = np.zeros(m - k)
        e1[0] = 1
        normx = np.linalg.norm(x)
        v = (sign(x[0]) * normx * e1) + x
        v = np.divide(v,np.linalg.norm(v))
        I = np.identity(x.size) - (2 * np.outer(v, np.conj(v).T))
        R[k:m,k:n] = np.matmul(I, R[k:m,k:n])
        W[k:m, k] = v
    return (W,R)

def form_q(W):
    Q = None
    #TODO
    (m,n) = W.shape
    P = np.identity(m).astype(complex)
    for c in range(m):
        for k in range(n):
            P[k:m, c] = ((P[k:m,c])-((np.outer(W[k:m,k],np.conj(W[k:m,k]).T))*2)@((P[k:m,c])))
    Q = np.conj(P).T
    return(Q)