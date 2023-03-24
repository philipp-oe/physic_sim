#######################################################################
##### Mertopolis Monte Carlo simulation of an argon gas/fluid #########
#######################################################################

import numpy as np
from scipy.constants import e,k
from matplotlib import pyplot as plt
from matplotlib import ticker
from tqdm import tqdm
from numba import njit

# constants
epsilon = 1.654e-21;
sigma = 3.405e-10;
m = 6.6335e-26;      # mass in kg
rho = 1400;          # density in kg/m^3
d = (m/rho)**(1/3);
temp = 95;
beta = 1/(temp*k);

# parameters
n = 8;
N = n**3;
box = n*d;

@njit
def lenard_jones(r):
    energy = 4*epsilon*( (sigma/r)**12 - (sigma/r)**6);
    return np.sum(energy)

@njit
def initi_lattice(n, d, cutoff = box/2):
    ''' 
    Initialises the grid as a simple cubic lattice

    Parameters: 
    - n: int number of atoms in 1 dimension
    - d: lattice parameter
    - cutoff: cut-off distance, until the atoms are includet in the calculations

    Returns:
    - positions: array (N,3), representing the 3 spacial coordinats of all atoms
    - energy: int, gives the initial energy of the system
    '''

    N = n**3;
    positions = np.zeros((N,3));
    count = 0;
    for i1 in range(n):
        for i2 in range(n):
            for i3 in range(n):
                positions[count] = d*i1, d*i2, d*i3;
                count += 1;

    # calcualte the pairwise distances of each atom            
    delta = positions[:, np.newaxis, :]- positions[np.newaxis, :, :];
    delta -= np.rint(np.round(delta,decimals=15)/box)*box;
    dist = (np.sqrt(np.sum(delta**2, axis=2)));
    dist[dist > cutoff] = 0;
    dist = dist[dist !=0];
    

    energy = (lenard_jones(dist))
    return positions, energy


def single_dist(pos, M, box_size, cutoff = box/2):
    delta = pos[np.newaxis, : , :] - pos[M];
    delta -= np.rint(np.round(delta,decimals=15)/box)*box;
    dist = (np.sqrt(np.sum(delta**2, axis=2)));
    dist[dist > cutoff] = 0;
    dist = dist[dist !=0];
    return dist


def initial_energy(positions, box_size, cutoff = box/2):

    delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :];

    # Apply periodic boundary conditions
    delta = np.where(delta > 0.5*box_size, delta - box_size, delta);
    delta = np.where(delta < -0.5*box_size, delta + box_size, delta);

    # Calculate distances
    dist = np.triu(np.sqrt(np.sum(delta**2, axis=2)));
    dist = dist.reshape(-1);
    #dist[dist > cutoff] = 0;
    dist = dist[dist !=0];

    return np.sum(lenard_jones(dist));

def metropolisMC(num_particles, num_steps, step_size):
    N = num_particles**3

    # initalise positions and energy
    #positions, energy = initi_lattice(num_particles, d);
    positions = box * np.random.rand(num_particles**3, 3); 
    energy = initial_energy(positions, box);

    acc_moves = 0;
    acc_positions = [];
    acc_energys = [];

    for i in tqdm(range(num_steps)):
        # choose index of random particle
        M = np.random.randint(0,N);

        move = np.random.uniform(low= -step_size, high= step_size, size=(3));

        new_positions = positions.copy();
        new_positions[M] += move;
        
        old_energy = lenard_jones( single_dist(positions, M, box));
        new_energy = lenard_jones( single_dist(new_positions, M, box));

        delta_energy = new_energy - old_energy;

        # Accept or reject the move based on the Metropolis criterion
        if delta_energy < 0 or np.random.uniform() < np.exp(-delta_energy*beta):
            positions[M] = new_positions[M];
            energy += delta_energy;
            acc_moves += 1;
            acc_positions.append(positions);

        # Append current position to list of accepted positions
        
        acc_energys.append(energy);

    # Compute acceptance ratio
    acceptance_ratio = acc_moves/num_steps;
    return acc_energys, acc_positions , acceptance_ratio;


Vpot, pos, acc_ratio = metropolisMC(n,int(1e6),0.0125*box);
Vpot = np.array(Vpot)
pos = np.array(pos)
np.save('positions.npy',pos)
np.save('energy.npy',Vpot)

print(acc_ratio)