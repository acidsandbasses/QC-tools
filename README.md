# qctools

Set of python scripts for further post-processing of quantum chemistry output files. Based on the [cclib](https://github.com/cclib/cclib) library and projected DOS code of [GaussSum](https://github.com/gausssum/gausssum).

## General fcc100 Surface in ASE

Requires [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/). This method provides a way to construct the Atoms object of a 100 surface on an fcc crystal by specifying arbitrary lattice vectors at the surface, rather than just specifying number of cells in x and y (and z) directions.

Can  be used in tandem with XYZ file parser.

Usage:
```
import numpy as np
from ase import Atoms
from ase.build import add_adsorbate
from ase.visualize import view

from qctools.xyz_file_parser import XYZ
from qctools.fcc100_general_shape import fcc100_general

# Import adsorbate from xyz file:
molecule = XYZ('filename.xyz')

# rotate molecule so mean square plane is parallel to x-y plane
molecule.orient_xy()

# rotate molecule about z
molecule.rotate('z', theta) # in radians

# make the atoms object
symbols, positions = molecule.split_symbols_positions()
molecule_ASE = Atoms(symbols = symbols,
                 positions = positions)
                 
# Make surface object
A = np.array([4, -3]) # Ax, Ay
B = np.array([3, 4]) # Bx, By

slab = fcc100_general('Cu', size=(A,B,3), vacuum=20) # size[2] is number of layers in surface slab

# add adsorbate
h = 3.5
add_adsorbate(slab, molecule_ASE, h, 'ontop')

# take a look
view(slab)
```
