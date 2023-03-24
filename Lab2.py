###################################################################
###### MD for 125...1000 argon atoms with PBC #####################
###################################################################

# import of all required packages
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.metrics import pairwise_distances
from scipy.constants import e,k
from tqdm import tqdm

# constants for the system
eps = 1.654e-21
sigma = 3.405e-10
m = 6.6335e-26      # mass in kg
Temp = 94.4              # temp in  K
rho = 1400          # density in kg/m^3

def LJ_force(r,box):
    ''' function to calculate the LJ-Potential and force for a given
    input vector of shape (N,3) and outputs the force F (shape(N,3))
    and the potential V as an int.'''
    N = len(r)
    Vpot = 0
    dist = r[:, None, :] - r[None, :, :]
    dist -= np.rint(np.round(dist,decimals=15)/box)*box
    r2 = np.triu(np.sum(dist**2,axis=2))
    r6 = r2**3
    r12 = r6**2
    with np.errstate(divide='ignore', invalid='ignore'):
        ff = 24*eps*(2*(sigma**12/r12) - (sigma**6/r6))[...,None] *dist/r2[...,None]
        force = np.nansum(ff, axis=1) - np.nansum(ff, axis=0)
        Vpot = np.nansum(4*eps*( (sigma**12/r12) - (sigma**6/r6)) )
    return force, Vpot  

def kinectic(velo,N,m):
    ''' func to calculate the kinetic energy and temperature with given 
    velocity as input. Kinetic energy and temperature are returned '''
    cst = (3*N*k)
    kin = .5*m*np.sum(np.sum(velo**2,axis=1))
    temp = 2*kin/cst
    return temp, kin

def md_integrate(r0, v0, dt, t_steps, box, velo_rescale=False, m =6.6335e-26):
    ''' molecular dynamics integration
        inputs: r0 initial positions (N,3)
                v0 initial velocities (N,3)
                dt timestep, t_steps # of time steps
                box size of the box
                velo_rescaling if velocity rescaling is needed
        outputs: r_steps positions (t_steps, N, 3)
                 v_steps velocities (t_steps, N, 3)
                 Kin, Vpot kinetic/potenital energy (t_steps, N)'''
    N = len(r0)
    r = r0.copy(); v = v0.copy()
    r_steps = np.zeros((t_steps,r0.shape[0],3))
    v_steps = np.zeros((t_steps,v0.shape[0],3))
    Kin = np.zeros(t_steps)
    Vpot = np.zeros(t_steps)
    a = np.zeros_like(r)

    r_steps[0] = r
    v_steps[0] = v
    _, Kin[0] = kinectic(v,N,m)
    _, Vpot[0] = LJ_force(r,box)
    for i in tqdm(range(1,t_steps)):
        am = a 
        r += v*dt + 0.5*am*dt**2
        F, V = LJ_force(r,box)
        a = F/m
        v += .5*(am+a)*dt
        T, K = kinectic(v,N,m)
        if velo_rescale and i< 1000 and i%50 ==0:
            v *= np.sqrt(Temp/T)
            _, K = kinectic(v,N,m)
        r_steps[i] = r
        v_steps[i] = v 
        Vpot[i] = V
        Kin[i] = K
    return r_steps, v_steps, Vpot, Kin
        
def init(n,d):
    '''Initial lattice setup process - setting all atoms in a simple cubic lattice'''
    N = n**3
    r = np.zeros((N,3))
    v = np.random.uniform(0,1,(N,3))-0.5
    count = 0
    for i1 in range(n):
        for i2 in range(n):
            for i3 in range(n):
                r[count] = d*i1, d*i2, d*i3
                count += 1
    sumv = np.sum(v, axis=0)/N
    sumv2 = np.sum(np.sum(v**2,axis=1))/N
    fs = np.sqrt(3*Temp/sumv2)
    v = (v-sumv)*fs
    return r, v

def coord_wrapping(r,box):
    ''' using coordinate wrapping to plot all atoms inside the given box'''
    return r - np.floor(np.round(r,decimals=15)/box)*box

##### TASK: Periodic BC ###################################### 
N = 2
n = 2 
d = 10.229*sigma/864**(1/3)
box = 7.5e-10
delt = 1e-15
T_max = 5e-12
t_steps = int(T_max/delt)
T_max = t_steps*delt
t = np.arange(T_max,step=delt)*1e12 # time in ps
x = np.zeros((N,3),dtype=float)
v = np.zeros((N,3),dtype=float)
x[1,0] = 4*1e-10
r, v, Vpot, E_kin = md_integrate(x,v,delt,t_steps,box)

x1 = r[:,0,0]
x2 = r[:,1,0]
v1 = v[:,0,0]
v2 = v[:,1,0]

plt.figure(figsize=(10.5, 7))
plt.rc('font',size=20)
plt.plot(t,x1*1e10, label='Atom 1')#, marker='x', ls='none')
plt.plot(t,x2*1e10, label='Atom 2')#, marker='x', ls='none')
plt.xlim(0, 5)
plt.xlabel(f'Time [ps]')
plt.ylabel(f'Distance [Å]')
plt.legend(loc=[.7,.6])
plt.savefig('x_coordinates.png')

plt.figure(figsize=(10.5, 7))
plt.rc('font',size=20)
plt.plot(t,E_kin/e*1e3, label='$\mathcal{V}$')
plt.plot(t,Vpot/e*1e3, label='$\mathcal{K}$')
plt.plot(t,(E_kin+Vpot)*1e3/e, label='$\mathcal{H}$')
plt.xlabel('Time $[ps]$')
plt.ylabel('Energy $[meV]$')
plt.xlim((0,5))
plt.legend()
plt.savefig('Energies.png')

plt.figure(figsize=(10.5, 7))
plt.rc('font',size=20)
plt.plot(t,v1, label='Atom 1')
plt.plot(t,v2, label='Atom 2')
plt.xlabel('Time $[ps]$')
plt.ylabel('Velocity $[a.u.]$')
plt.xlim((0,5))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=3, fancybox=True, shadow=True)
plt.savefig('x_velocities.png')


##### Initial conditions for 125...1000 atoms ##########################
n = 10
N = n**3
d = (m/rho)**(1/3)
box = n*d
delt = 1e-15
T_max = 5e-12
t_steps = int(T_max/delt)
T_max = t_steps*delt
t = np.arange(T_max,step=delt)*1e12 # time in ps

r0, v0 = init(n,d)

r, v, Vpot, Kin = md_integrate(r0, v0, delt, t_steps, box, velo_rescale=True)

plt.figure(figsize=(10.5,7))
plt.rc('font',size=20)
plt.plot(t,Kin/e, label='$\mathcal{K}$')
plt.plot(t,Vpot/e, label='$\mathcal{V}$')
plt.plot(t,(Kin+Vpot)/e, label='$\mathcal{H}$')
plt.xlabel('Time $[ps]$')
plt.ylabel('Energy $[meV]$')
plt.xlim((0,5))
plt.legend()
plt.savefig('Energies_'+str(N)+'.png')

r_wrap = coord_wrapping(r,box)
for i in [0,250,1000,2500,4000]:
    plt.figure(figsize=(10.5,7))
    plt.rc('font', size=20)
    plt.scatter(r_wrap[i,2*n**2:3*n**2,1]*1e10, r_wrap[i,2*n**2:3*n**2,2]*1e10, s=30, c='navy', alpha=0.5)
    plt.axvline(x=box*1e10, linestyle='--', linewidth=1.0, color='grey')
    plt.axvline(x=0, linestyle='--', linewidth=1.0, color='grey')
    plt.plot([-2,box*1e10+2],[0,0], linestyle='--', linewidth=1.0, color='grey')
    plt.plot([-2,box*1e10+2],[box*1e10,box*1e10], linestyle='--', linewidth=1.0, color='grey')
    plt.ylim(-1,box*1e10+1)
    plt.xlim(-1,box*1e10+1)
    plt.xlabel('y-Coordinate [Å]')
    plt.ylabel('z-Coordinate [Å]')
    plt.savefig('pos_t'+str(i)+'_N'+str(N)+'.png')
