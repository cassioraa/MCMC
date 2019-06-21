# Autor: Cássio Roberto de Andrade Alves
# Data: junho/2019
# Doutorando em economia (FEA-RP)

# coding: utf-8

# # Método de Monte Carlo

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from numba import jit

# Distribuições 
from numpy.random import uniform as U
from numpy.random import normal as N



# ### Exemplo 1
# 
# Suponha que $X \sim N(0,1)$ e que queremos calcular a probabilidade de X ser não negativo, isto é, $P(X>0)$. Em termos de integração de Monte Carlo, isto é equivalente a avaliar a seguinte integral:
# 
# \begin{equation}
# E[I_{x>0}(x)]=\int_{-\infty}^{\infty}I_{x>0}(x)f_N(x|\mu=0, \sigma^2=1)dx,
# \end{equation}
# em que $I_{x>0}(x)$ é a função indicadora. A aproximação usando o MC fica:
# 
# \begin{equation}
# E[I_{x>0}(x)] = \frac{1}{N} \sum_{i=1}^N I_{x>0}(x).
# \end{equation}
# 
# Como conhecemos a distribuição normal, temos que esperar que o resultado da integral seja aproximadamente $0.5$.
# 
# 

# In[74]:


def I(x):
    if x>0:
        y=1
    else:
        y=0
    return y


# In[75]:


np.random.seed(4321)
def monte_carlo_normal(Niter):
    #Niter = 10000
    y = np.zeros((Niter,))
    z = 0
    for i in range(Niter):
        x = N(0,1)
        
        if x>0:
            y[i]=1
        else:
            y[i]=0
        

        z += y[i]
    integral = z/Niter
    
    return integral, y 
qe.util.tic()
integral2,y = monte_carlo_normal(10000)
qe.util.toc()

integral2   


# In[97]:


np.random.seed(4321)
@jit(nopython=True)
def monte_carlo_normal(Niter):
    #Niter = 10000
    y = np.zeros((Niter,))
    z = 0
    for i in range(Niter):
        x = np.random.normal(0,1)
        
        if x>0:
            y[i]=1
            a=1
        else:
            y[i]=0
            a=0
        

        z += a
    integral = z/Niter
    
    return integral, y 
qe.util.tic()
integral2,y = monte_carlo_normal(10000)
qe.util.toc()

integral2   


# In[77]:

Niter = 10000
MR2  = np.cumsum(y)/range(1,Niter+1)
std2 = np.sqrt(np.cumsum((y-MR2)**2))/range(1,Niter+1)


# In[78]:


fig, ax = plt.subplots(figsize=(10,3))

ax.set_ylim(np.mean(y)-30*std2[Niter-1], np.mean(y)+30*std2[Niter-1])
ax.set_xlim(-250,Niter)
plt.plot(MR2,color='k', label="Médias recursivas")
plt.plot(MR2+2*std2, color="purple", lw=0.5, alpha=0.5, label="$\pm$ 2 desvios-padrão")
plt.plot(MR2-2*std2, color="purple", lw=0.5, alpha=0.5)
plt.axhline(0.5, color='r', lw=0.5, alpha=1, label="0.5")
plt.legend()
ax.tick_params(direction="in")
ax.set_xlabel("Iteração, i")
plt.show()
#plt.savefig('/home/cassio/Documents/Doutorado/USP-RP/Disciplinas/Estatística Aplicada/MCMC/text/fig1.pdf')
