import numpy as np

import os
import sys
import subprocess

import cclib
from cclib.parser.utils import PeriodicTable

"""
Usage:

$ python xy-plane.py [filename]

filename can be an output file (.log or .out) or .xyz file

"""


class XYZ():
	"""
	Class to contain and manipulate xyz coordinates of a molecule

	Usage:

	initialize with either 
	
	1)
	mol = XYZ()
	mol.get_coords(filename)

	2) or directly
	mol = XYZ(filename)

	currently supported are any ".log" files that can be parsed with cclib, or ".xyz" files

	mol.coords is a list of lists which mimics the layout of an xyz file
	[ ['C1', np.array(x1, y1, z1)], 
	  ['C2', np.array(x2, y2, z2)], ...]	

	orient_xy() orients the molecule so that it is centered at (0,0,0) 
				and the mean-square plane is along the x-y plane.

	write_xyz_file(fname) writes current coordinates to file

	lower level operations:

	rotate(axis, angle)
	translate(v) translates coordinates r -> r + v
	shift_origin(new_origin) translates coordinats r -> r - <r> + new_origin


	"""

	# -----------
	# Functions
	# -----------	
	

	def lowest_evec(self, evals, evecs):
		# Find lowest eigenvalue
		# corresponding eigenvector is the normal vector defining the least squares plane
		i_smallest_eval = 0
		smallest_eval = evals[0]
		for i in range(len(evals)):
			if evals[i] < smallest_eval:
				smallest_eval = evals[i]
				i_smallest_eval = i
		n = evecs[:, i_smallest_eval] / np.sqrt(np.dot(evecs[:, i_smallest_eval], evecs[:, i_smallest_eval]))
		# Make positive component in z direction
		if n[2] >= 0:
			return n
		else:
			return (-1)*n 	

	def mean_plane_normal(self, mol_xyz):
		# Covariance matrix
		cov = np.zeros((3,3))
		for mol in mol_xyz:
			cov += np.outer(mol[1],mol[1])
		# Eigenvectors
		l, v = np.linalg.eig(cov)
		# Return the eigenvector with lowest eigenvalue
		return self.lowest_evec(l, v)	

	def write_xyz_file(self, filename):
		xyz_file = open(filename, 'w')
		xyz_file.write("{:s} \n{:s}".format(str(len(self.coords)), filename))
		for mol in self.coords:
			line = "\n" + mol[0]
			for coord in mol[1]:
				line = line + "\t {:.6f}".format(coord)
			xyz_file.write(line)
		xyz_file.close()	

	def rotate(self, axis, theta):
		if axis == "x":
			Rot = np.array([[1,             0,                  0],
		                    [0, np.cos(theta), (-1)*np.sin(theta)],
		                    [0, np.sin(theta),      np.cos(theta)]])
		if axis =="z":
			Rot = np.array([[np.cos(theta), (-1)*np.sin(theta), 0],
		                    [np.sin(theta),      np.cos(theta), 0],
		                    [            0,                  0, 1]])
		for mol in self.coords:
			mol[1] = np.dot(Rot, mol[1])

	def get_coords(self, fname):
		"""
		Returns a list like
			# [
			# ['Zn', array([0,0,0])],
			# ['N', array([1,1,1])],...
			# ]	

		"""
		ftype = fname.split(".")[-1]
		mol_xyz = []
		if ftype=="log" or ftype=="out":
			opfile = cclib.io.ccread(fname)
			pt = PeriodicTable()
			cnt = 1
			for anum, acoords in zip(opfile.atomnos,opfile.atomcoords[-1]):
				print("cnt {0}: anum: {1} acoords: {2}".format(cnt, anum, str(acoords)))
				mol_xyz.append([pt.element[anum], acoords])
				cnt += 1
		elif ftype=="xyz":
			with open(fname) as xyzfile:
				line = xyzfile.readline()
				cnt = 1
				while line:
					# Actions for current line
					if cnt>2:
						asym = line.split()[0]
						acoords = np.array(line.split()[1:],dtype=float)
						mol_xyz.append([asym, acoords])
					line = xyzfile.readline()
					cnt+=1
		return mol_xyz

	def get_center(self):
		X0, Y0, Z0 = 0,0,0
		for mol in self.coords:
			X0 += mol[1][0]
			Y0 += mol[1][1]
			Z0 += mol[1][2]
		return np.array([X0/self.Natoms, Y0/self.Natoms, Z0/self.Natoms])

	def translate(self, v = np.zeros(3)):
		for mol in self.coords:
			mol[1] = mol[1] + v

	def shift_origin(self, new_origin = np.zeros(3)):
		self.translate(v = (-1)*self.get_center() + new_origin)

	
	def orient_xy(self):
		"""
		1) shift molecule so center is at 0
		2) compute mean square plane (msp)
		3) rotate so msp = xy plane
		"""

		# 1) 
		self.shift_origin()
		self.write_xyz_file(self.fname.split(".")[-0]+"-shifted.xyz")

		# 2)
		Nv = self.mean_plane_normal(self.coords)
		theta = np.arccos(Nv[2])
		phi = np.arctan(Nv[1]/Nv[0])

		# Rotate about z axis so plane normal is in yz plane
		self.rotate("z", np.pi/2-phi)
		# Rotate about y axis so normal plane is xy plane
		self.rotate("x", (-1)*theta)

		self.write_xyz_file(self.fname.split(".")[-0]+"-rotated.xyz")


	def __init__(self, filepath=None):
		# Initial Coordinates:
		if filepath == None:
			self.coords = []
		if filepath != None:
			# Parse XYZ file
			self.fname = os.path.basename(filepath) # strip off the file name from the directory
			t = os.path.dirname(filepath) # get directory part of file name
			if t:
				os.chdir(t) # if we are not already in that directoy, cd into it
			self.coords = self.get_coords(self.fname)

		self.Natoms = len(self.coords)



# -------------
# Main Program
# -------------

#

#

if __name__ == "__main__":

	molecule = XYZ(sys.argv[1])

	molecule.orient_xy()

