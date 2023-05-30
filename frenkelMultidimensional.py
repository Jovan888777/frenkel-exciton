# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 22:21:34 2023

@author: Jovan Markovic
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import *

elemwiseConj = np.vectorize(lambda z: complex(z.real, -z.imag))


def modulo(z):
    return (z.real**2 + z.imag**2)**(1/2)


def getBra(Psi):
    return elemwiseConj(Psi.T)


hbar = 0.06351


def chromophores(Xe, K, E, M, v, X, Psi, t, s, n, m):
    if n != len(Xe) or n != len(K) or n != len(M) or n != len(X):
        return "Wrong dimensions in number of modes n!"
    
    if m != len(Xe[0]) or m != len(K[0]) or m != len(M[0]) or m != len(X[0]) or m != len(E) or m != len(Psi):
        return "Wrong dimensions in number of molecules m!"
    
    dt = t / s
    
    
    Psi1ModNullY = []
    Psi2ModNullY = []
    tX = []
    eTotY = []
    x11Y = []
    mechEY = []
    excitonEY = []
    Norm = []
    IPR = []
    
    XPrev = X.copy()
    
    for step in range(s):
        Psi1ModNullY.append(abs(Psi[0][0]))
        Psi2ModNullY.append(abs(Psi[1][0]))
        tX.append(step*dt)
        x11Y.append(X[0][0])
        Norm.append(sum(abs(Psi[i][0])**2 for i in range(len(Psi))))
        IPR.append(1 / sum(abs(Psi[i][0])**4 for i in range(len(Psi))))
        # WARNING: Each step, simulation constructs H again, instead of updating. Inefficient.
        
        H = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if j == i:
                    Eps_n = E[j][0] - ((K[:, j:j+1] * Xe[:, j:j+1]).T @ X[:, j:j+1])[0][0]
                    H[i, j] = Eps_n
                    
                #1D Case (linear esemble/thin monolayer)
                '''
                try:
                    if j == i + 1:
                        H[i, j] = v
                except:
                    pass
                try:
                    if j == i - 1:
                        H[i, j] = v
                except:
                    pass
                '''
                
                
                #2D Case (square monolayer)
                try:
                    if j == i + 1 and (i + 1) % int(m**(1/2)) != 0:
                        H[i, j] = v
                except:
                    pass
                try:
                    if j == i - 1 and i % int(m**(1/2)) != 0:
                        H[i, j] = v
                except:
                    pass
                
                try:
                    if j == i - int(m**(1/2)):
                        H[i, j] = v
                except:
                    pass
                try:
                    if j == i + int(m**(1/2)):
                        H[i, j] = v
                except:
                    pass
                
        
        mechE = [M[i][j] / 2 * ((X[i][j] - XPrev[i][j]) / dt)**2 + K[i][j] / 2 * X[i][j]**2 for i in range(n) for j in range(m)]
        sumMechE = sum(mechE) * 0.01036
        mechEY.append(sumMechE)
        excitonE = (getBra(Psi) @ H @ Psi)[0][0] * 0.01036
        excitonEY.append(excitonE)
        eTotY.append(excitonE + sumMechE)
        
        K1 = dt*complex(0, -1)/hbar*(H @ Psi)
        K2 = dt*complex(0, -1)/hbar*(H @ (Psi + K1 / 2))
        K3 = dt*complex(0, -1)/hbar*(H @ (Psi + K2 / 2))
        K4 = dt*complex(0, -1)/hbar*(H @ (Psi + K3))
        Psi = Psi + K1 / 6 + K2 / 3 + K3 / 3 + K4 / 6
        
        #Psi = Psi / (getBra(Psi) @ Psi)[0][0].real**(1/2)
        
        PsiAug = np.tile(Psi, (1, n))
        A = -(K / M)*(X - abs(PsiAug.T)**2*Xe)
        
        XPrevDummy = X.copy()
        X = 2*X - XPrev + A*dt**2
        XPrev = XPrevDummy
        
        if step == s // 5:
            print("20% done!")
        if step == 2 * s // 5:
            print("40% done!")
        if step == 3 * s // 5:
            print("60% done!")
        if step == 4 * s // 5:
            print("80% done!")
        
    return [[tX, [eTotY, Norm, Psi1ModNullY, Psi2ModNullY, x11Y, excitonEY, mechEY, IPR]], [[i for i in range(len(Psi))], [abs(Psi[i][0]) for i in range(len(Psi))]]]
    
    #plt.plot(tX, excitonEY)
    #plt.savefig("plot.pdf")
    #plt.show()
    
 
# Potential pursuit of numerical determination of fixed points for Psi
'''
xA, xB, Psi1, Psi1S, Psi2, Psi2S = symbols('xA, xB, Psi1, Psi1S, Psi2, Psi2S')
eq1 = Eq(100*Psi1 + 100*xA*Psi1 + Psi2, 0)
eq2 = Eq(100*Psi2 + 100*xB*Psi2 + Psi1, 0)
eq3 = Eq(100*Psi1S + 100*xA*Psi1S + Psi2S, 0)
eq4 = Eq(100*Psi2S + 100*xB*Psi2S + Psi1S, 0)
eq5 = Eq(xA - Psi1S*Psi1, 0)
eq6 = Eq(xB - Psi2S*Psi2, 0)

sol = nsolve([eq1, eq2, eq3, eq4, eq5, eq6], [xA, xB, Psi1, Psi1S, Psi2, Psi2S], [1, 1, 1, 1, 1, 1])
#soln = [tuple(v.evalf() for v in s) for s in sol]

print(sol)
'''


'''
The simulation uses GROMACS units:
https://manual.gromacs.org/documentation/2019/reference-manual/definitions.html
Typical ranges:

x_e in [~0.01, ~100]
m about ~10 as its C + H
E about 10eV ~ 100
v about 0.1eV ~ 1
k in [~1, ~1e7]
'''
    
# Simple dummy data: dimer, one mode
'''
Xe = np.array([[1, 1]])
K = np.array([[100, 100]])
E = np.array([[100], [100]])
M = np.array([[10, 10]])
v = 1
X0 = np.array([[0, 0]])
Psi0 = np.array([[2**(1/2)/2 + 0.01], [2**(1/2)/2 - 0.01]])

chromophores(Xe, K, E, M, v, X0, Psi0, 100, 100000, 1, 2)
'''



# Dummy data, dimer, 3 modes
'''
Xe = np.array([[3, 3],
               [2, 2],
               [1, 1]])
K = np.array([[100, 100],
              [200, 200],
              [300, 300]])
E = np.array([[100], 
              [100]])
M = np.array([[10, 10],
              [10, 10],
              [10, 10]])
v = 10
# Sample X0 from Gaussian
X0 = np.array([[0, 0],
               [0, 0],
               [0, 0]])
Psi0 = np.array([[1], [0]])

res = chromophores(Xe, K, E, M, v, X0, Psi0, 20, 300000, 3, 2)
'''

# Old Ardy data, olympicene, too many modes for modest simulation!
'''
f = open("freqs.csv")

rawText = f.read()

f.close()

lines = rawText.split("\n")
dataLines = lines[3:len(lines) - 1]
dataMatrix = [e.split(',') for e in dataLines]
# Most modes omega <= 50 and xe <= 10
dataMatrix = [e for e in dataMatrix if float(e[1]) <= 5 and float(e[3]) <= 20]
print(len(dataMatrix))


M = np.array([[10, 10] for i in range(len(dataMatrix))])
K = np.array([[float(dataMatrix[i][1])**2*M[i][0], float(dataMatrix[i][1])**2*M[i][1]] for i in range(len(dataMatrix))])
Xe = np.array([2*[float(dataMatrix[i][3])] for i in range(len(dataMatrix))])
E = np.array([[100], [100]])
v = 10
X0 = np.array([[0, 0] for i in range(len(dataMatrix))])
Psi0 = np.array([[1], [0]])


res = chromophores(Xe, K, E, M, v, X0, Psi0, 0.1, 100000, len(dataMatrix), 2)
'''

# New Sexithiophene model. GREAT DATA!
m = 36
kB = 0.0083144621
    

M = np.array([[1 for i in range(m)],
              [1 for i in range(m)]])
K = np.array([[300**2 for i in range(m)],
              [200**2 for i in range(m)]])
Knull = np.array([[0 for i in range(m)],
                  [0 for i in range(m)]])
Xe = np.array([[0.05 for i in range(m)],
               [0.01 for i in range(m)]])
E = np.array([[300] for i in range(m)])
v = 70
X0 = np.array([[np.random.normal(0, kB*300/300**2) for i in range(m)],
               [np.random.normal(0, kB*300/200**2) for i in range(m)]])
Psi0 = np.array([[1]] + [[0] for i in range(m - 1)])

res = chromophores(Xe, K, E, M, v, X0, Psi0, 4, 200_000, 2, m)



def plotFn(X, Y, xLab='Unlabeled', yLab='Unlabeled'):
    plt.plot(X, Y, color='red')
    plt.xlabel(xLab)
    plt.ylabel(yLab)
    #plt.savefig("plot.pdf")
    plt.show()

print('Done! :)')