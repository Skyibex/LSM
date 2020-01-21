
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  

T      =   1000                    # total simulation length [ms]
dt     =   0.1                     # step size [ms]
time   =   np.arange(0, T+dt, dt)  # step values [ms]

vr = 30
Ne = 800
Ni = 200
re = np.random.uniform(0,1,(Ne, 1))
ri = np.random.uniform(0,1,(Ni, 1))

a  = np.append(0.02*np.ones((Ne,1)),    0.02 +0.08*ri)
b  = np.append(0.2*np.ones((Ne,1)),     0.25 -0.05*ri)
c  = np.append( -65+15*(re**2.0),    -65*np.ones((Ni,1)))
d  = np.append(8.0 - 6*re**2 , 2*np.ones((Ni,1))  )

S                    = np.zeros((Ne+Ni, Ne+Ni))
S[0:Ne+Ni, 0:Ne]     = 0.5*np.random.uniform(0,1,(Ne+Ni,Ne))
S[0:Ne+Ni, Ne:Ne+Ni] =   - np.random.uniform(0,1,(Ne+Ni,Ni))

v0 = -65.0*np.ones((Ne+Ni,1))
u0 =  v0*np.reshape(b,(len(b),1))
v  = np.zeros((Ne+Ni, len(time))) 
u  = np.zeros((Ne+Ni, len(time))) 

v[:,0] = v0.T
u[:,0] = u0.T

firing = []
for t in range(1, T):
    I =  np.append( 5*np.random.normal(size=(Ne, 1)),
                    2*np.random.normal(size=(Ni, 1)))    
    fired     = np.where(v[:,t-1] >= vr)[0]
    fire_data = {'times': t + 0*fired, 'neurons': fired}
    fire_data = pd.DataFrame(fire_data)
    firing.append(fire_data)
    v[fired, t-1] = c[fired]
    u[fired, t-1] = u[fired,t-1] + d[fired]
   
    vT= v[:,t-1]
    uT= u[:,t-1]
    I = I + np.sum(S[:,fired], axis=1)
    vT     = vT + (0.04 * vT**2 + 5*vT  + 140 - uT + I)*0.5
    vT     = vT + (0.04 * vT**2 + 5*vT  + 140 - uT + I)*0.5
    uT     = uT +  a * (b * vT - uT)
    v[:,t] = vT
    u[:,t] = uT
           
firing = pd.concat(firing)
firing.sort_values(by=['times'])
firing = np.array(firing)
plt.plot(firing[:,1], firing[:,0], '.', markersize=1.1)

