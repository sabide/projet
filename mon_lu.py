import numpy as np
from scipy.linalg import lu
import time



def lu_fact(A):
  n,m = np.shape(A) # verfier si n==m
  U = np.copy(A) # je copie A dans upper
  L = np.eye(n) # identité pour L

  for k in range(n-1):
      pivot = U[k,k]
      for i in range(k+1,n):
          L[i,k] = U[i,k] / pivot
          for j in range(k+1,n):
              U[i,j] = U[i,j] - L[i,k]*U[k,j]

  for i in range(n):
      for j in range(i):
          U[i,j]=0


  return L, U  

# pour tester sur un exemple à la main
#A=np.array( [ [1.,2.,1. ] ,[2.,2.,1.],[1.,1.,1.]]  ) 

n=200
print("LU Scipy (lapack)")
A=np.random.rand(n,n)
start_time = time.time()
P,L,U=lu(A) # factorisation
tcpu = time.time() - start_time # temps
delta=np.abs(A - P @ L @ U) # verification
err=delta.max() # calcul erreru
print("%d %e %e" % (n,tcpu,err))


print("MON LU")
A=np.random.rand(n,n)
start_time = time.time()
L,U=lu_fact(A)
tcpu = time.time() - start_time # temps
delta=np.abs(A - L @ U) # verification
err=delta.max() # calcul erreru
print("%d %e %e" % (n,tcpu,err))
