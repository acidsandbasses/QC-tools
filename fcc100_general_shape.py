import numpy as np

from ase import Atoms
from ase.build import fcc100, add_adsorbate
from ase.data import reference_states, atomic_numbers

def fcc100_general(symbol, a=None, size=None, vacuum=None, alignx=True):
    """
    Constructs supercell outlined by vectors A and B

    size is a tuple or list of length 3
    """
    
    A = size[0] 
    B = size[1]
    layers = size[2]
    
    Z = atomic_numbers[symbol]

    if a is None:
        #sym = reference_states[Z]['symmetry']
        #if sym != structure:
        #    raise ValueError("Can't guess lattice constant for %s-%s!" %
        #                     (structure, symbol))
        a = reference_states[Z]['a']
    
    M = np.linalg.inv(np.transpose(np.stack((A,B))))
      
    # Brute force - check all integer points (n,m) to see if they are inside of the supercell
    # keep the transformed point M^(-1).(n,m) rather than (n,m)
    inside_points = [] 
    ns = np.arange(np.min((0, A[0], B[0], (A+B)[0])), np.max((0, A[0], B[0], (A+B)[0]))+1)
    ms = np.arange(np.min((0, A[1], B[1], (A+B)[1])), np.max((0, A[1], B[1], (A+B)[1]))+1)
    for n in ns:
        for m in ms:
            p = np.array([n,m])
            t = M.dot(p)
            if (t[0] >= 0) & (t[0] < 1):
                if (t[1] >= 0) & (t[1] < 1):
                    inside_points.append(t)
                    # inside_points.append(p)
    pos_AB = np.array(inside_points)
    
    # construct positions for atoms in the supercell
    
    positions = np.zeros((layers, len(pos_AB), 3)) # z, xy, 3
    
    positions[..., 0] = pos_AB[:, 0].reshape(1, -1)
    positions[..., 1] = pos_AB[:, 1].reshape(1, -1)
    positions[..., 2] = np.arange(layers).reshape(-1, 1)/(layers-1)
    
    positions[-2::-2, ..., :2] += M.dot(np.array([0.5, 0.5]))
    positions[..., :2] = positions[..., :2] % 1
    
    #print(positions)
    
    tags = np.empty((layers, len(pos_AB)), int)
    tags[:] = np.arange(layers, 0, -1).reshape((-1, 1))
    
    slab = Atoms(np.ones(len(positions.reshape(-1,3))) * Z,
                 tags = tags.ravel(),
                 pbc=(True, True, False))
    
    slab.set_positions(positions.reshape(-1, 3))
    
    slab.set_cell([np.concatenate((A, np.zeros(1)))*a*np.sqrt(0.5),
                   np.concatenate((B, np.zeros(1)))*a*np.sqrt(0.5),
                   np.array([0, 0, layers-1])*a*0.5],
                   scale_atoms=True)
    
    if vacuum is not None:
        slab.center(vacuum, axis=2)
    
    if 'adsorbate_info' not in slab.info:
        slab.info.update({'adsorbate_info': {}})
    
    slab.info['adsorbate_info']['cell'] = np.transpose(np.stack((A,B)))*a*np.sqrt(0.5)
    slab.info['adsorbate_info']['sites'] = {'ontop': (0.0, 0.0), 'hollow': tuple(M.dot(np.array([0.5, 0.5]))), 'bridge': tuple(M.dot(np.array([0.5, 0])))}
    
    return slab