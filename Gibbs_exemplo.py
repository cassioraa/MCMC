# Autor: Cássio Roberto de Andrade Alves
# Data: junho/2019
# Doutorando em economia (FEA-RP)

# ver: http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-9.html
# ver: http://www.dme.ufrj.br/mcmc/
# coding: utf-8

#TODO: usar o jit nas funções para otimizar o código 

import numpy as np
import scipy.special as ssp
import matplotlib.pyplot as plt
import quantecon as qe

np.random.seed(123) # fixa semente

# dados
y = np.array([4,5,4,1,0,4,3,4,0,6,3,3,4,0,2,6,3,3,5,4,5,3,1,4,4,1,5,5,3,4,2,5,2,2,3,4,2,1,3,2,2,
      1,1,1,1,3,0,0,1,0,1,1,0,0,3,1,0,3,2,2,0,1,1,1,0,1,0,1,0,0,0,2,1,0,0,0,1,1,0,2,3,3,
      1,1,2,1,1,1,1,2,4,2,0,0,0,1,4,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1])

n = len(y)

# Gráfico dos dados
fig, ax = plt.subplots(figsize=(15,5))

ax.plot(range(1851, 1851+n), y ,"o-", color="k", label="$y$")
ax.plot(range(1851, 1892), np.mean(y[0:40])*np.ones(1892-1851), "--", color="r", label="Média")
ax.plot(range(1891, 1851+n), np.mean(y[41:n])*np.ones(1851+n-1891), "--", color="r")
ax.tick_params(direction="in")
plt.axvline(1891, color="green", label="Ano de 1891")
plt.legend()
plt.savefig("text/fig3.pdf")
plt.show()

# função densidade condicional total pi(m|lambda, phi)
def cond_total_m(m,lambbda,phi,y,n,alpha,beta,gamma,delta):

    if m==0:
        sm = 0
    else:
        sm = sum(y[0:m])
        
    if n==0:
        sn = 0
    else:
        sn = sum(y[0:n])
        
    out = lambbda**(alpha + sm -1) * np.exp(-(beta+m)*lambbda) * phi**(gamma+sn-sm-1) * np.exp(-(delta+n-m)*phi)
    return out


# valor inicial para os parâmetros
m = 112
lambbda = 3
phi=0.9

# valor para os hiperparâmetros
alpha = 0.001
beta  = 0.001
delta = 0.001
gamma = 0.001

# Amostra de cond_total_m
def random_m(lambbda,phi,y,n,alpha,beta,gamma,delta):
    totais = np.zeros(n)
    for i in range(n):
        totais[i] = cond_total_m(i, lambbda,phi,y,n,alpha,beta,gamma,delta)
    totais = totais/sum(totais)
    
    return int(np.random.choice(range(n), size=1, p = totais))

#===========================#
# == Amostrador de gibbs == #
#===========================#
np.random.seed(1234) # fixa semente
# Passo 1)

j, Niter = 1, 5000

m = 41


draws = np.zeros((3,Niter))

qe.util.tic()

while j<Niter+1:
    if j%500==0:
        print("Iteração ", j)

    sm = sum(y[0:m])
    sn = sum(y)
    
    # Passo 2)
    
    lambbda = np.random.gamma(alpha+sm, 1/(beta+m))            # bloco 1
    phi     = np.random.gamma(gamma+sn-sm, 1/(delta+n-m))      # bloco 2
    m       = random_m(lambbda,phi,y,n,alpha,beta,gamma,delta) # bloco 3
    
    draws[0,j-1] = np.copy(lambbda)
    draws[1,j-1] = np.copy(phi)
    draws[2,j-1] = np.copy(m)
    
    # passo 3) 
    j += 1
qe.util.toc()

#========================================================================
# Gráfico dos valores amostrado e distribuição (histograma) das amostras
#========================================================================

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15,10))

labels = ["$\\lambda$","$\\phi$","$m$"]

for i in range(2):
    ax[i,0].plot(draws[i,:], color="k", label=labels[i])
    ax[i,1].hist(draws[i,:], color="grey", alpha= 0.5, bins=30, density=True, label=labels[i])
    ax[i,1].axvline(np.mean(draws[i,:]), linestyle="--", color="k", label="Média")
    ax[i,0].tick_params(direction="in")
    
    ax[i,0].tick_params(direction="in")
    ax[i,1].tick_params(direction="in")
    
    
    ax[i,0].legend()
    ax[i,1].legend()
    
    

x, yy = np.unique(draws[2,:], return_counts=True)

dic = dict(zip(x+1851, yy))
ax[2,0].plot(draws[2,:] + 1851, color="k", label=labels[2])
ax[2,1].bar(dic.keys(), dic.values(), width=0.1, color="grey", alpha=0.5,  ec="grey", label="$m$")
ax[2,1].axvline(np.mean(draws[2,:]+1851) , linestyle="--", color="k", label="Média")


ax[2,0].tick_params(direction="in")
ax[2,1].tick_params(direction="in")

ax[2,0].legend()
ax[2,1].legend()

plt.savefig("text/fig4.pdf")

# densidade gamma
# $p(x) = x^{k-1}\frac{e^{-x/\theta}}{\theta^k\Gamma(k)}
def gammapdf(x, k, theta):

	
    c = (theta**k) * ssp.gamma(k)
    y = x**(k-1) * np.exp(-x/theta)
    
    return y/c

#gammapdf(1,2,1/3)

#================================
# Constrói a trajetória do Gibs
#================================

N=50
m = int(np.mean(x))


lambdas = np.linspace(min(draws[0,:]),max(draws[0,:]), 50)
phis    = np.linspace(min(draws[1,:]),max(draws[1,:]), 50)

joint = np.zeros((N,N))

a = sum(y[0:m]) + alpha
b = m + beta
c = sum(y) - sum(y[0:m]) + gamma
d = n - m + delta

for i in range(N):
    for j in range(N):
        joint[i,j] = gammapdf(lambdas[i], a, 1/b) * gammapdf(phis[j], c,1/d)


joint = joint/Niter


xss = np.zeros((2*Niter,2))

xss[0,:] = draws[0:2,0]
xss[1,1] = draws[1,1]
xss

for i in range(2,Niter):
    xss[2*(i-1)-1,0] = draws[0,i-2]
    xss[2*i-2,0] = draws[0,i-1]
    
    xss[2*(i-1),1] = draws[1, i-1]
    xss[2*i-1,1] = draws[1,i]
xss[:5,0]


fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
ax[0].contourf(lambdas,phis,joint, cmap="BuGn")
ax[0].plot(xss[0:11,0],xss[0:11,1], color="k")
ax[1].contourf(lambdas,phis,joint, cmap="BuGn")
ax[1].plot(xss[0:100,0],xss[0:100,1], color="k")

ax[0].tick_params(direction="in")
ax[1].tick_params(direction="in")

ax[0].set_ylabel("$\phi$")
ax[0].set_xlabel("$\lambda$")
ax[0].set_title("$5$ iterações")


ax[1].set_ylabel("$\phi$")
ax[1].set_xlabel("$\lambda$")
ax[1].set_title("$50$ iterações")
plt.savefig("text/fig5.pdf")
plt.show()


#=============================
# Tamanho da amostra efetivo
#=============================

from statsmodels.tsa import stattools 

# Calcula o tamanho efetivo
neff = np.zeros(3)
for i in range(3):
    fac = stattools.acf(draws[i,:], nlags=20)
    neff[i] = Niter/(1+2 * sum(fac[1:]))

print("neff \n", neff)

#=========================================
# Gráfico da autocorrelação
#=========================================
from statsmodels.tsa import stattools 
from matplotlib.ticker import MaxNLocator

index = list(range(0,len(fac)))

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,8))

for i in range(3):
    Y = draws[i,:]
    fac  = stattools.acf(Y, nlags=20)
    index = 1+np.array(range(len(fac)))
    
    ax[i].bar(np.array(index), fac, width=0.5, color='grey', label=labels[i])
    ax[i].axhline(0, color='red', lw=0.5)
    ax[i].axhline((1.96/np.sqrt(len(Y))), color='blue', linestyle = "--", lw=0.5)
    ax[i].axhline(-(1.96/np.sqrt(len(Y))), color='blue', linestyle = "--", lw=0.5)
    ax[i].legend(loc="upper right")
    #ax[i].set_xlim(0.5,20)
    
    ax[i].tick_params(direction="in")
    ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[i].set_ylabel("$\\rho_j$")
ax[2].set_xlabel("Defasagens, $j$")
plt.savefig("text/fig6.pdf")
    
plt.show()