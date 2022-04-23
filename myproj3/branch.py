import numpy as np 
import time 
adjMatrix = list() 
distMat = list()

def initDict(N): 
    D = dict()
    A = np.random.uniform(size=N*2) 
    A.shape = (2,N)
    for L in A:
        L = [str(round(e,3)).ljust(5) for e in L] #Distance calculated using euclidian distance 
    for c2 in range(A.shape[1]):
        for c1 in range(c2):
            x1,y1 = A[:,c1]
            x2,y2 = A[:,c2]
            D[(c1,c2)] = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return A,D

def repr(L,dist):
    rL = [' '.join([str(n) for n in L]) + ' '] 
    rL.append(str(round(dist,3)))
    return '\n'.join(rL)

def generateRandom(N,D):
    L = np.arange(N) 
    np.random.shuffle(L)
    return repr(L,getTotalDist(L,D))

def getTotalDist(L,D,n=None): 
    d =0
    for first,next in zip(L[:-1],L[1:]): 
        if first < next:
            d += D[(first,next)] 
        else:
            d += D[(next,first)] 
        if n and d > n:
            return None 
        return d
    
def branchBound(): 
    A,D = initDict(N) 
    distMAt = []
    for i in range(N):
        a = []
        for j in range(N):
         a.append(0) 
        distMat.append(a)
    for a in range(N): 
        for b in range(N):
            for k in sorted(D.keys()):
                if (a == k[0]) and (b == k[1]):
                    x=k
                    distMat[a][b] = distMat[b][a] = D[k] 
    for i in range(5):
        result = generateRandom(N,D) 
    print(distMat)
    cost = 0
    for row in range(N):
        for col in range(row+1, N): 
            cost += distMat[row][col]
    print("Cost of the Network Topology : "+str(cost))
    
if __name__=="__main__": 
    print ("Enter the value of N") 
    N = int(input()) 
    branchBound()