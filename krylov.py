import numpy as np
from scipy.linalg import solve_triangular
from numpy.linalg import norm

def cg(A, b, tol=1e-12):
    m = A.shape[0]
    x = np.zeros(m, dtype=A.dtype)
    r_b = [1]
    # todo
    r = np.matmul(A,np.transpose(x)) - b
    p = r.copy()

    for k in range(m):
      ä = np.dot(np.transpose(r),r) / np.dot(np.dot(np.transpose(p),A),p)
      x = x + ä * p
      r = r - ä * A * p
      ß = np.dot(np.transpose(r),r) / (np.transpose(r_b[k-1]) * r_b[k-1])
      p = r + ß * p

      r_b.append(np.linalg.norm(r) / np.linalg.norm(b))
      if (r_b[k-1] < tol):
        break

    return x, r_b


def arnoldi_n(A, Q, P):
    m, n = Q.shape
    q = np.zeros(m, dtype=Q.dtype)
    h = np.zeros(n + 1, dtype=A.dtype)
    #todo
    s = solve_triangular(P,A)
    v = s @ Q[:, -1]
    for i in range(n):
      h[i] = np.conjugate(np.transpose(Q[:, i])) @ v
      v = v - h[i] * Q[:, i]
    h[n] = np.linalg.norm(v)
    if h[n] != 0:q = v / h[n]
    else: 0
    return h, q


def gmres(A, b, P=None, tol=1e-5):
    if P is None:
        # default preconditioner P = I
        P = np.identity(A.shape[0])

    x = np.zeros(A.shape[1])
    r_b = [1]
    m = A.shape[0]

    for i in range(m):
        Q = np.zeros((m, 1), dtype=b.dtype)
        ß = solve_triangular(P, b)
        Q = ß / np.linalg.norm(ß)
        h = [[0]]
        h_temp = np.zeros((i + 2, i + 1), dtype=A.dtype)
        h_temp[:i, :i] = h[i]
        h1, q1 = arnoldi_n(A, Q, P)
        h_temp[i + 1, i] = h1[i]
        h = h_temp

        q_temp = np.zeros((m, i + 2), dtype=A.dtype)
        q_temp[:, :i + 1] = Q[i]
        q_temp[:, i + 1] = q1[i]
        Q = q_temp

        ql, rl = np.linalg.qr(h)
        e = np.zeros(h.shape[0])
        e[0] = 1

        y = solve_triangular(rl, np.linalg.norm(np.conjugate(np.transpose(q[i]))) * np.linalg.norm(ß) * np.linalg.norm(e))

        r = np.matmul(h, y) - np.linalg.norm(ß, 2) * e
        x = np.matmul(Q, y)

        r_b.append(np.linalg.norm(r, 2) / np.linalg.norm(ß, 2))

        if r_b[-1] < tol:
            break

    return x, r_b