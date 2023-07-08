######################################################################
##### 2D Ising Model with long range interations #####################
######################################################################
'''
code runs the simulation of a 2D Ising model for different inter-
action leghts - the interation lenght is given by the order of 
neibors k and the distance norm used for that is the manhatten-distance
the interation parameter J = 1 so all results are in a.u.
Simulations are done for a fixed temperature
'''

import numpy as np
from matplotlib import pyplot as plt
#from tqdm import tqdm
from numba import njit
import time
from matplotlib import animation
import random
import math

#np.random.seed(42069) 
#np.random.seed(2147483648)
np.random.default_rng()

random_seed = np.random.get_state()[1][0]
print(random_seed)

# Const.
J = 1;
path = '/Users/philipp/Documents/Sthlm Uni/Simulation Methods/'

def initi_lattice(L):
    ''' initial random lattice / cold start lattice'''
    return np.random.choice([-1,1], size=(L,L));

def plot(lattice):
    plt.figure(figsize=(9,8))
    plt.imshow(lattice, cmap='gray',vmin=-1,vmax=1);

@njit
def energy(lattice, i, j, k):
    '''
    calculates the energy for a given spin i,j in the lattice,
    within neighbor order k
    in: lattice (N,N)
        i,j position in lattice (int)
        k neighbor order (int)

    out: energy (double)
    '''
    energy = 0
    L = lattice.shape[0]
    
    for m in range(i-k, i+k+1):
        for n in range(j-k, j+k+1):
            if m == i and n == j:
                continue
            dx = abs(m - i)
            dy = abs(n - j)
            #dist = dx + dy      # manhatten distance norm
            dist = np.sqrt(dx**2 + dy**2) # eulcidian distance norm
            if dist > k:
                continue
            energy += J* lattice[i,j] * lattice[m%L, n%L] / dist**2
    return -energy

@njit
def total_energy(lattice,k):
    '''
    calculates the total energy of the entire lattice
    in: lattice (N,N)
        k neighbor order (int)

    out: en energy (double)
    '''
    size = lattice.shape[0];
    en = 0
    for i in range(size):
        for j in range(size):
            en += energy(lattice,i,j,k)
    return en

@njit
def magnetesation(lattice):
    '''
    calculates the magnetization of the system
    in: lattice (N,N)
    out: magnetisation (double)
    '''
    return np.abs(np.sum(lattice))

@njit
def metropolisMC(lattice, num_steps, k, temp):
    '''
    Monte Carlo Simulation for the 2D Ising model
    in: lattice (N,N)
        num_steps max # of MC steps (int)
        k neighbor order (int)
        temp temperature of the sytem (double)

    out: energys energy over MC-steps (num_steps,1)
         acc_spins accepted lattice configurations (num_steps,N,N)
         acc_ratio acceptance ratio of MC algo (double)
         mag magnetization over MC-steps (num_steps,1)
    '''
    L = lattice.shape[0]
    acc_moves = 0;
    acc_spins = [];
    energys = [];
    mag = [];

    en = total_energy(lattice,k)
    energys.append(en)
    mag.append(magnetesation(lattice))
    old_energy = en;

    for i in (range(num_steps)):
        n = np.random.randint(0,L);
        m = np.random.randint(0,L);
        prop_lattice = lattice.copy()

        # choose random spin in lattice and flip it
        prop_lattice[n,m] *= -1;
        prop_energy = energy(prop_lattice,n,m,k);

        delta_energy = -old_energy + prop_energy;

        if delta_energy < 0 or np.random.random() < np.exp(-delta_energy/temp):
            #lattice[n,m] *= -1;
            lattice = prop_lattice;
            en += delta_energy;
            old_energy = en;
            acc_moves += 1;

            acc_spins.append(lattice)
            #energys.append(en)
            #mag.append(magnetesation(lattice))
    acc_ratio = acc_moves/num_steps

    return energys, acc_spins, acc_ratio, mag  

def animate(i):
    ax.clear()
    ax.imshow(Spin[i],cmap='gray', vmin=-1, vmax = 1)

###### FIRST PART: STUDY CHANGE OF SYSTEMS ##############################
# prameters
sizes = np.array([30,50]);
T= 1 
k_order = np.array([1, 2, 5]);

#np.random.seed(2506)
L = 150

# takes the same start lattice for all neighbor iterations
lattice = initi_lattice(L);


#precompile metropolisMC
_,_,_,_ = metropolisMC(lattice, 1000, 1, 1) 
print('compile done')

for k in (k_order):
    start_spins = lattice.copy()
    _, spins, acc_ratio, _ = metropolisMC(start_spins, int(0.75e6), k, T)  
    if acc_ratio == 0:
        print('Failed for k= ',k)
        continue
    print(acc_ratio)
    counter = 0;
    num_plot = np.floor(np.array([0, 0.4, 0.85])*len(spins)).astype(int)
    for i in num_plot:
        plot(spins[i])
        plt.rc('font', size=20)
        plt.savefig('lat_L'+str(L)+'_k'+str(k)+'_'+str(counter)+'.png')
        counter +=1

    np.save(path+'spins_'+str(k)+'_L'+str(L)+'.npy' , np.array(spins))
    
    # gives animations of the change of the lattice
    print('pre - animation')
    Spin = np.array(spins)
    c = int(100/k)
    print(c)
    Spin = Spin[::c]
    fig, ax = plt.subplots(figsize=(9,8))
    ani = animation.FuncAnimation(fig, animate, frames=Spin.shape[0], interval=600, blit=False)
    ani.save('/Users/philipp/Documents/Sthlm Uni/Simulation Methods/spins_L'+str(L)+'_k'+str(k)+'.mp4', writer='ffmpeg', fps=48, bitrate=1800)
    print('done k= ',k)
print(L, ' done')

#plt.show()
