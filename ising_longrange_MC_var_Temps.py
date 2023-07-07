######################################################################
##### 2D Ising Model with long range interations #####################
######################################################################
'''
code runs the simulation of a 2D Ising model for different inter-
action leghts - the interation lenght is given by the order of 
neibors k and the distance norm used for that is the manhatten-distance
the interation parameter J = 1 so all results are in a.u.
Simulations are done for various temperature.
Tested for 50x50 and 30x30 lattices
'''
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from numba import njit
import time 

# Const.
J = 1;

def initi_lattice(L):
    ''' initial random lattice / cold start lattice'''
    return np.random.choice([-1,1], size=(L,L));

def plot(lattice):
    plt.figure()
    plt.imshow(lattice, cmap='gray',vmin=-1,vmax=1);
    plt.colorbar();

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
            #dist = dx + dy
            dist = np.sqrt(dx**2 + dy**2)
            if dist > k:
                continue
            energy += lattice[i,j] * lattice[m%L, n%L] / dist**4
    energy *= J
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

def drop(array,percent):
    size = int(len(array)*percent)
    return array[size:]


@njit
def metropolisMC(lattice,num_steps, k, temp):
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
    #acc_spins = [];
    energys = [];
    mag = [];

    en = total_energy(lattice,k)
    #energys.append(en)
    #mag.append(magnetesation(lattice));

    for i in (range(num_steps)):
        n = np.random.randint(0,L);
        m = np.random.randint(0,L);
        prop_lattice = lattice.copy()

        old_energy = energy(lattice,n,m,k);

        # choose random spin in lattice and flip it
        prop_lattice[n,m] *= -1;
        prop_energy = energy(prop_lattice,n,m,k);

        delta_energy = prop_energy - old_energy;

        if delta_energy < 0 or np.random.random() < np.exp(-delta_energy/temp):
            lattice = prop_lattice;
            en += delta_energy;
            acc_moves += 1;
            

        if i < int(0.7*num_steps):
            continue

        energys.append(en)
        mag.append(magnetesation(lattice))
     
    acc_ratio = acc_moves/num_steps

    return energys, lattice, acc_ratio, mag  

###################################################################

'''# not used anymore
@njit
def avg(lattice,k,T):
    size = lattice.shape[0]
    mag = 0
    en = 0
    Z = 0
    for i in range(size):
        for j in range(size):
            E = energy(lattice,i,j,k)
            exp = np.exp(-E/T)
            en += E*exp
            mag += lattice[i,j]#*exp
            Z += exp
    
    en /= Z
    mag /= Z
            
    return en, mag
'''


###################################################################
'''
Analysis of magnetization and energy over various temperatures for 
different neighbor orders 
'''
temps = np.linspace(0.5, 6.5, 50)
L = 125
k_order = np.array([1, 2, 5])


start_spins = initi_lattice(L)
en = np.zeros((4,len(temps)))
mag = np.zeros((4,len(temps)))
sus = np.empty((4,len(temps)))

max_temps = len(temps)

print(f'\nStarting computation for L={L}')
for j,k in enumerate(k_order):
    tic = time.time()
    spin = start_spins
    print('0.00%',end='')
    for i,T in enumerate(temps):
        ee, spin, ratio, mm = metropolisMC(spin,int(.75e6/np.sqrt(k)),k,T)
        E = np.mean(ee)
        M = np.mean(mm)
        en[j,i] = E
        mag[j,i] = M
        #sus[j,i] = (np.mean(np.square(mm))- M**2)/T
        print(f'\r{(i/max_temps*100):.2f}%', end='')

    toc = time.time()
    print(f'\rk={k} done in {np.round(toc-tic,2)} s')

en /= L**2
mag /= L**2

plt.figure(figsize=(10,7))
plt.rc('font', size=20)
plt.plot(temps,en[0],'o:', label='k=1')
plt.plot(temps,en[1],'o:', label='k=2')
plt.plot(temps,en[2],'o:', label='k=5')
#plt.plot(temps,en[3],'o:', label='k=10')
plt.xlabel(r'T [a.u.]')
plt.ylabel(r'$\bar{\mathcal{H}}$/N [a.u.]')
plt.legend()
plt.savefig('/Users/philipp/Documents/Sthlm Uni/Simulation Methods/energy_'+str(L)+'.png')

plt.figure(figsize=(10,7))
plt.rc('font', size=20)
plt.plot(temps,mag[0],'o:', label='k=1')
plt.plot(temps,mag[1],'o:', label='k=2')
plt.plot(temps,mag[2],'o:', label='k=5')
#plt.plot(temps,mag[3],'o:', label='k=10')
plt.xlabel(r'T [a.u.]')
plt.ylabel(r'$\bar{M}/N$ [a.u.]')
plt.legend()
plt.savefig('/Users/philipp/Documents/Sthlm Uni/Simulation Methods/Mag_'+str(L)+'.png')

plt.figure(figsize=(10,7))
plt.rc('font', size=20)
plt.plot(temps,sus[0],'o:', label='k=1')
plt.plot(temps,sus[1],'o:', label='k=2')
plt.plot(temps,sus[2],'o:', label='k=5')
#plt.plot(temps,mag[3],'o:', label='k=10')
plt.xlabel(r'%\chi [a.u.]%')
plt.ylabel(r'$\bar{M}/N$ [a.u.]')
plt.legend()

plt.show()
random_seed = np.random.get_state()[1][0]
print(random_seed)