######################################################################
##### Analysis of the results of the metropolis Monte Carlo sim ######
######################################################################

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
#from scipy.spatial.distance import pdist
from scipy.constants import e,k, N_A
from tqdm import tqdm

# load data from metropolis Monte Carlo simulation
positions = np.load('positions.npy');
energys = np.load('energy.npy');
x = np.arange(0,len(energys));

# paramesters
n = 5; 
N = n**3;
Temp = 95;
epsilon = 1.654e-21;
sigma = 3.405e-10;
m = 6.6335e-26;      # mass in kg
rho = 1400;          # density in kg/m^3
d = (m/rho)**(1/3);
temp = 95;
beta = 1/(temp*k);
box = n*d;

# plot of energy
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0,0))

plt.figure(figsize=(9,6))
plt.rc('font', size=20)
plt.plot(x, energys/e)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xlim(0,500000)
plt.ylim(-8,-6)
plt.yticks(np.linspace(-8, -6, 5))
plt.xlabel(f'# of MC Iterations')
plt.ylabel(f'Energy [eV]')
plt.savefig('figure/energy.png')


# adjust the data to drop the values at start of simulation
energys = energys[25000:];
positions = positions[25000:];

avg_energy = np.average(energys);
err_energy = np.sqrt(np.var(energys));
print('Energy: '+str(avg_energy/e)+ ' ± '+str(err_energy/e)+ ' eV');

Cv = (3*k*N/2) + ((err_energy)**2/(k*Temp**2));
print(Cv/e);

##### RDF Calculations  ###############################################

def RDF(dist,box, num_bins):
    dr = box/num_bins
    rs = dr*np.arange(num_bins)
    hist = np.zeros(num_bins, dtype=int)
    
    # Flatten the distance matrix and remove duplicate distances
    dist = np.linalg.norm(dist, axis=2).flatten()

    # Sort the distances in ascending order and eliminating all 0 elements
    dist = dist[dist != 0]
    dist = np.sort(dist, kind='heapsort')

    # Compute the bin edges
    bin_edges = np.linspace(0, box, num_bins+1)
    
    bin_idx = -1
    for j, dist_el in enumerate(dist):
        while dist_el > bin_edges[bin_idx+1]:
            bin_idx += 1
        hist[bin_idx] += 1

    # Normalize the histogram to get the radial distribution function
    rdf = box**3/(N*np.pi*dr*rs**2)*hist*2/N

    return rdf, bin_edges

def block_average(data, n_blocks):
    """
    Computes the block average of a 1D numpy array.

    Parameters
    ----------
    data : numpy.ndarray
        The input data to be block-averaged.
    block_size : int
        The size of the blocks over which to average the data.

    Returns
    -------
    numpy.ndarray
        The block-averaged data.
    """
    # Calculate the number of blocks.
    block_size = len(data) // n_blocks

    # Trim the data so that the length is an exact multiple of the block size.
    data = data[:n_blocks * block_size]

    # Reshape the data into blocks.
    blocks = data.reshape((n_blocks, block_size))

    # Compute the mean of each block.
    block_means = np.mean(blocks, axis=1)

    return block_means

# to speed up the computation 
positions = positions[::10];
num_bin = 100;
g = [];

print(positions.shap)
for i in tqdm(range(positions.shape[0])):
    r = positions[i];
    dist = r[:, np.newaxis, :] - r[np.newaxis, :, :];
    dist -= np.rint(np.round(dist,decimals=15)/box)*box;
    hist, bin_edge = RDF(dist, box, num_bin);
    g.append(hist);

print(bin_edge.shape)

g = np.array(g);
np.save('g.npy',g);
np.save('edges.npy',bin_edge);

g = np.load('g.npy')
bin_edges = np.load('edges.npy')
bins = g.shape[1]
avg_g = np.zeros(bins)
for i in tqdm(range(bins)):
    data = g[:,i];
    avg_g[i] = np.average(data);

plt.figure(figsize=(9,6))
plt.rc('font', size=20)
plt.plot(bin_edges[:-1]*1e10,avg_g)
plt.xlabel('radial distance [$\AA$]')
plt.ylabel('RDF [a.u.]')
plt.savefig('figure/avg_RDF.png')
plt.show()