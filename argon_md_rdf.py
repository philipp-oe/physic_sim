###################################################################
###### MD for 125...1000 argon atoms with PBC #####################
###################################################################

# import of all required packages
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.constants import e,k
from tqdm import tqdm
import matplotlib.animation as animation

# constants for the system
eps = 1.654e-21
sigma = 3.405e-10
m = 6.6335e-26      # mass in kg
Temp = 95              # temp in  K
rho = 1400          # density in kg/m^3

def LJ_force(r,box):
    ''' function to calculate the LJ-Potential and force for a given
    input vector of shape (N,3) and puts out the force F (shape(N,3))
    and the potential V as an int.'''
    N = r.shape[0]
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
    
    return force, Vpot, dist 


def RDF(dist,box, num_bins):
    dr = box/num_bins
    rs = dr*np.arange(num_bins)
    hist = np.zeros(num_bins, dtype=int)
    
    # Flatten the distance matrix and remove duplicate distances
    dist = np.linalg.norm(dist,axis=2).flatten()

    # Sort the distances in ascending order and eliminating all 0 elements
    dist = np.sort(dist, kind='heapsort')
    dist = dist[dist != 0]
    jjdist = dist.copy()
    # Compute the bin edges
    bin_edges = np.linspace(0, box, num_bins+1)

    bin_idx = -1
    for j, dist in enumerate(dist):
        while dist > bin_edges[bin_idx+1]:
            bin_idx += 1
        hist[bin_idx] += 1

    # Normalize the histogram to get the radial distribution function
    hist = hist*2/N
    rdf = box**3/(N*np.pi*dr*rs**2)*hist*2/N

    return rdf, bin_edges


def kinectic(velo,N,m):
    ''' func to calculate the kinetic energy and temperature with given 
    velocity as input. Kinetic energy and temperature are returned '''
    cst = (3*N*k)
    kin = .5*m*np.sum(np.sum(velo**2,axis=1))
    temp = 2*kin/cst
    return temp, kin

def md_integrate(r0, v0, dt, t_steps, box, velo_rescale=False, m =6.6335e-26):
    num_bin = 100
    N = len(r0)
    r = r0.copy(); v = v0.copy()
    r_steps = np.zeros((t_steps,r0.shape[0],3))
    v_steps = np.zeros((t_steps,v0.shape[0],3))
    Kin = np.zeros(t_steps)
    Vpot = np.zeros(t_steps)
    Ts = np.zeros(t_steps)
    g = np.zeros((t_steps,num_bin))
    a = np.zeros_like(r)

    r_steps[0] = r
    v_steps[0] = v
    Ts[0], Kin[0] = kinectic(v,N,m)
    _, Vpot[0], dist = LJ_force(r,box)
    #g[0], bin_edges = RDF(dist,box, num_bin)
    for i in tqdm(range(1,t_steps)):
        am = a 
        r += v*dt + 0.5*am*dt**2
        F, V, dist = LJ_force(r,box)
        #rdf, _ = RDF(dist,box, num_bin)
        a = F/m
        v += .5*(am+a)*dt
        T, K = kinectic(v,N,m)
        if velo_rescale and i< 1000 and i%50 ==0:
            v *= np.sqrt(Temp/T)
            T, K = kinectic(v,N,m)
        r_steps[i] = r
        v_steps[i] = v 
        Vpot[i] = V
        Kin[i] = K
        Ts[i] = T
        #g[i] = rdf
    return r_steps, v_steps, Vpot, Kin, Ts#, g, bin_edges
        
def init(n,d):
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
    v = (v-sumv)#*fs
    return r, v

def coord_wrapping(r,box):
    return r - np.floor(np.round(r,decimals=15)/box)*box

# TODO: find avg's of Vpot, Kin and T, partially needed for c_v
##### Initial conditions for 125...1000 atoms ##########################
n = 8
N = n**3
d = (m/rho)**(1/3)
box = n*d
delt = 5e-15
T_max = 20e-12
t_steps = int(T_max/delt)
T_max = t_steps*delt
t = np.arange(T_max,step=delt)*1e12 # time in ps

r0, v0 = init(n,d)

#r, v, Vpot, Kin, Temps, g, bin_edges = md_integrate(r0, v0, delt, t_steps, box)
r, v, Vpot, Kin, Temps= md_integrate(r0, v0, delt, t_steps, box, velo_rescale=True)

t = t[:-1]

plt.figure(figsize=(9,8))
plt.rc('font', size=20)
plt.plot(t,Kin/e, label='$\mathcal{K}$')
plt.plot(t,Vpot/e, label='$\mathcal{V}$')
plt.plot(t,(Kin+Vpot)/e, label='$\mathcal{H}$')
plt.xlabel('Time $[ps]$')
plt.ylabel('Energy $[eV]$')
plt.xlim((0,20))
plt.legend()
plt.savefig('figure/Energies_'+str(N)+'.png')

plt.figure(figsize=(9,8))
plt.plot(t,Temps, label='Temp.')
plt.xlabel('Time $[ps]$')
plt.ylabel('Temperature $[K]$')
plt.xlim((0,20))
plt.savefig('figure/Temperature_'+str(N)+'.png')

# saving all important vars to do the plotting and rdf in an other file

np.save('positions.npy',r)
np.save('velocities.npy',v)
np.save('potential.npy',Vpot)
np.save('kinetic.npy',Kin)
np.save('temperature.npy',Temps)
#np.save('rdfs.npy', g)
#np.save('bin_edges.npy', bin_edges)

'''plt.figure(figsize=(10.5,7))
plt.plot(bin_edges[:-1], g[25,:])
#plt.plot(bin_edges[:-1], g[250,:])
plt.plot(bin_edges[:-1], g[50,:])
#plt.plot(bin_edges[:-1], g[750,:])
plt.plot(bin_edges[:-1], g[100,:])
plt.xlabel('distance')
plt.ylabel('RDF')
plt.show()'''

'''g = g[::10, :]
fig, ax = plt.subplots(figsize=(10.5,7))
def animate(i):
    ax.clear()
    ax.plot(bin_edges[:-1],g[i,:]) 

ani = animation.FuncAnimation(fig, animate, frames=g.shape[0], interval=60, blit=False)
#plt.show()
ani.save("rdf.mp4", writer='ffmpeg', fps=30, bitrate=1800)'''

'''for i in [0,100,500,1000,4000,8000,16000]:
    plt.figure(figsize=(9,8))
    plt.rc('font', size=20)
    plt.plot(bin_edges[:-1]*1e10,g[i,:])
    plt.xlabel('radial distance [$\AA$]')
    plt.ylabel('RDF [a.u.]')
    plt.savefig('figure/RDF_N'+str(N)+'_t'+str(i)+'.png')'''

