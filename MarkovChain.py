# Autor: Cássio Roberto de Andrade Alves
# Data: junho/2019
# Doutorando em economia (FEA-RP)

# coding: utf-8

# # 1. Cadeias de Markov

# ## 1.1 Espaço de estados discreto e finito

# In[116]:


import numpy as np
import matplotlib.pyplot as plt


# ## 1.2 Espaço de estados finito e enumerável

# Um exemplo é o passeio aleatório

# In[160]:


def gerar_rw(p,q,r, N):
    y = np.zeros(N)

    for t in range(N-1):

        u = np.random.uniform(0,1)

        if u<q:
            y[t+1] = y[t]-1
        if q<u<q+r:
            y[t+1] = y[t]
        if u>q+r:
            y[t+1] = y[t]+1
    return y


# In[176]:


p = 0.55
q = 0.45
r = 0
N = 500
y = gerar_rw(p,q,r,N)

p1 = 0.5
q1 = 0.5
r1 = 0
z = gerar_rw(p1,q1,r1,N)


# In[177]:


fig, ax = plt.subplots(figsize=(10,7))

ax.plot(y, "k", lw=0.7)
ax.plot(z, "g", lw=0.7)

ax.tick_params(direction="in")
plt.savefig("RW.pdf")
plt.show()