import time

from numba import njit
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_alg


L = 12
Nup = 6
Ndown = 6
t = -2.0
U = 10.0
hoppings = [(i, j, t) for i in range(L) for j in range(L) if abs(i - j) == 1]

@njit
def hammingWeight(n):
    res = 0
    for i in range(32):
        if n & (1 << i):
            res += 1
    return res

@njit
def getStates():
    spinUpStates = [i for i in range(1 << L) if hammingWeight(i) == Nup]
    spinDownStates = [i for i in range(1 << L) if hammingWeight(i) == Ndown]
    states = [(spinUp << L) + (spinDown) for spinUp in spinUpStates for spinDown in spinDownStates]
    return states

@njit
def doubleOccNum(state):
    spinUpState = state >> L
    spinDownState = state & ((1 << L) - 1)
    return hammingWeight(spinUpState & spinDownState)


def Hamiltonian(states):
    n = len(states)
    H = sparse.lil_matrix((n,n))
    stateIdxMap = {}
    for i in range(n):
        stateIdxMap[states[i]] = i
    for i in range(n):
        H[i, i] = U * doubleOccNum(states[i])
        for a, b, t in hoppings:
            for s in range(2):
                if (states[i] & (1 << (s * L) << b)) and not (states[i] & (1 << (s * L) << a)):
                    state = states[i] ^ (1 << (s * L) << b) ^ (1 << (s * L) << a)
                    j = stateIdxMap[state]
                    H[j, i] = t
    return H

if __name__ == '__main__':
    start = time.time()
    states = getStates()
    end = time.time()
    print("Constructing State Cost = %s" % (end - start))

    start = time.time()
    H = Hamiltonian(states)
    end = time.time()
    print("Constructing Hamiltonian Cost = %s" % (end - start))

    start = time.time()
    E,V=sparse_alg.eigsh(H.tocsr(),1,which='SA')
    end = time.time()
    print("Solve EigenValue Cost = %s" % (end - start))
    print(E)
    










