# Usage:
# First argument: output file
# Second argument: starting energy of DOS spectrum (integer)
# Third argument: last energy of DOS spectrum (integer)
# Fourth argument: FWHM of gaussian fit

import sys
import os.path
import argparse
#import pandas as pd

import numpy as np
from scipy import linalg as LA

import matplotlib.pyplot as plt
import cclib
from cclib.parser.utils import PeriodicTable

from xsec import cross_secs

# ==================
# ----- Groups -----
# ==================

def readAtomGroupsFile(filename):
    """Read group info from a file.
    
    File format:
     First line must be one of "orbitals","atoms","allorbitals","allatoms".
     The remainder of the file is ignored in the case of allatoms/allorbitals.
     Otherwise a series of groups must be described with a name by itself on one line
     and a list of group members on the following line.
     As an example:
         atoms <--- grouptype
         Phenyl ring <--- groupname
         1,4-6,8 <--- atoms
         The rest <--- groupname
         2,3,7

    parses into dictionary
    """
    grp_file = open(filename, 'r')

    grouptype = next(grp_file).strip() # atoms, orbitals, etc ...
    while not grouptype:
        grouptype = next(grp_file).strip()

    atomgroups = {}
    
    for line in grp_file:
        
        if not line.strip():
            continue # ignores blank lines

        groupname = line.strip()
        atoms = []
        line = next(grp_file)
        
        parts = line.split(',')
        for x in parts:
            temp = x.split('-')

            if len(temp) == 1:
                atoms.append(int(temp[0]))

            else:
                atoms.extend(
                    range(int(temp[0]), int(temp[1]) + 1)
                    )

        # each item in the dictionary is a list of atoms in the group
        atomgroups[groupname] = atoms 

    grp_file.close()

    return atomgroups

def inferAtomGroups(atomnos):
    """
    Make groups of like atoms in the molecule

    Takes input list of atomic numbers 
    """
    atomgroups = {}
    pt = PeriodicTable()

    for i, anum in enumerate(atomnos):
        asym = pt.element[anum]

        if asym in list(atomgroups.keys()):
            atomgroups[asym].append(i+1)

        else:
            atomgroups[asym] = [i+1]

    return atomgroups

def convertAtomGroupsToOrbitals(atomgroups, atombasis):
    """
    Converts list of atoms to list of indices over atomic orbitals on those atoms
    
    eg:
    {
    'C': [1,2,3,4,5,6],
    'H': [7,8,9,10,11,12]
    }


    """
    groups = {}

    for k, v in atomgroups.items():
        groups[k] = []
        for i in v:
            groups[k].extend(atombasis[i-1])

    return groups

def verifyGroups(groups, aonames):
    """
    Checks that every atom in the molecule is present in a group
    and that no atom is present twice
    """

    g_all = []
    
    for x in groups.values():
        g_all.extend(x)

    g_all.sort()

    status = ''

    ok = ( g_all == list(range(len(aonames))))

    if not ok:
        status = 'Groups must contain every atom exactly once'

    return status

def makeGroups(data, filename = None, verbose = False):
    """
    Pipeline for either 
    1) making groups from a file (must be in same directory as output file)
    2) or computing groups based on like atoms
    """
    if verbose:
        print("Looking for file %s " % filename)

    if os.path.isfile(filename):      
        
        if verbose:
            print(" --- %s ---" % filename)
            t = open(filename)
            print(t.read())
            print(" ----------")
            t.close()

        atomgroups = readAtomGroupsFile(filename)

    else:

        if verbose:
            print('making groups of like atoms')
        
        atomgroups = inferAtomGroups(data.atomnos)

    groups = convertAtomGroupsToOrbitals(atomgroups, data.atombasis)

    status = verifyGroups(groups, data.aonames) 
    if status:
        print(status)
        sys.exit()

    return groups

# =====================
# ----- Functions -----
# =====================

def gaussian(x, mean, std):
    """
    returns gaussian function with mean/std evaluated on x values
    """
    return (1/np.sqrt(2*np.pi*std**2))*np.exp(-(x-mean)**2/(2*std**2))


def sum_gaussians(MOs, energies, fwhm, weights=None):
    """
    Sums gaussian functions centered at each of the energies 

    assumes energies is a 1D np array
    i.e. call this function separately for alpha/beta contributions
    """
    std = fwhm/(2*np.sqrt(2*np.log(2)))
    
    # makes exactly evenly spaced points inclusive
    
    s = np.zeros(len(energies))
    
    if weights is None:
        weights = np.ones_like(MOs)

    for i in range(len(MOs)):
        s += gaussian(energies, MOs[i], std)*weights[i]

    return s

    # write to output file "DOS_total.txt"
    #total_DOS_file = open("DOS_total.txt", "w")
    #total_DOS_file.write("Energy (eV)\tTotal DOS")
    #for x,y in zip(xvalues,DOS):
    #    total_DOS_file.write("\n" + str(x) + "\t" + str(y))


def weighted_sum_gaussians(MOs, energies, fwhm, weights):
    """
    Sums gaussian functions centered at each of the energies

    Optional weights matrix of dimension (# MOs) x (# groups)

    This function assumes:
       energies is 1D numpy array
       weights is 2D numpy array 

    i.e. call this function separately for alpha/beta contributions
    """
    
    s = []

    for i in range(np.shape(weights)[1]):
        s.append(
            sum_gaussians(MOs, energies, fwhm, weights = weights[:,i])
            )

    p = np.array(s)
    s.append(np.sum(p, axis=0))

    return s


def make_MPA(data):
    """
    Mulliken Population Analysis:

    returns contribution matrix or MPA matrix
    i = MO number
    a = AO number
    [c^T]_(ia) = [C^T]_(ia) [C^T S^T]_(ia)
    """
    return [x * np.dot(x, data.aooverlaps) for x in data.mocoeffs]

def make_LPA(data):
    """
    Lowdin Population analysis
    """
    e, _ = LA.eig(data.aooverlaps)
    evals = np.diag(e)
    sqrtS = np.real(LA.sqrtm(data.aooverlaps))

    return [(x @ sqrtS) * (x @ sqrtS) for x in data.mocoeffs]

def popAnalysis(data, poptype = 'MPA'):
    """
    return contribution matrix based on population analysis type

    mulliken or loewdin
    """
    if (poptype == 'Mulliken') | (poptype == 'MPA'):
        return make_MPA(data)

    if (poptype == 'Lowdin') | (poptype == 'Loewdin') | (poptype == 'LPA'):
        return make_LPA(data)


def sumOverGroups(contrib_matrix, groups):
    """
    contribution matrix is of size (# MOs) x (# AOs)

    returns matrix of size (# MOs) x (# groups) 
    by summing columns that are part of the same group

    these are the weights to be used in PDOS calculations
    """
    t = list(np.zeros_like(contrib_matrix))

    t = [
    np.zeros(
        (contrib_matrix[0].shape[0], len(groups))
        )
    for x in contrib_matrix
    ]

    i = 0
    for gname, gindices in groups.items():
        for j in gindices:
            t[0][:, i] += contrib_matrix[0][:, j]
            if len(t) == 2:
                t[1][:, i] += contrib_matrix[1][:, j]
        i += 1

    return t


def cleanShell(aoname):
    """
    input

    Orca style: Zn1_3DX2Y2
    GAMESS style:

    Convert to Zn_d
    """

    # pt = PeriodicTable()

    # a = pt.element[aonumber]
    s = aoname.split('_')[-1]

    if 'S' in s:
        return 's'

    if ('P' in s) or (len(s) == 1):
        return 'p'

    if ('D' in s) or (len(s) == 2):
        return 'd'

    if ('F' in s) or (len(s) == 3):
        return 'f'

def applyCrossSections(contrib_matrix, data, cross_secs, photon):
    """
    method:

    contrib_matrix[mo,ao]*xsec[ao]

                [     ]                      photons['XPS']=0
    cross_secs[cleanShell(data.aonames[ao])][photons[photon]]
    """
    new = list(np.zeros_like(contrib_matrix))

    pt = PeriodicTable()

    for mo, _ in enumerate(data. aonames):
        for i, n in enumerate(data.atomnos):
            asym = pt.element[n]
            for ao in data.atombasis[i]:
                s = cleanShell(data.aonames[ao])
                x = cross_secs[n][s][photon]

                new[0][mo,ao] = contrib_matrix[0][mo,ao] * x
                
                if len(new) == 2:
                    new[1][mo,ao] = contrib_matrix[1][mo,ao] * x
    
    """
    for mo in range(len(data.aonames)):
        for ao in range(len(data.aonames)):
            a, s = cleanShell(data.aonames[ao])
            new[0][mo,ao] = contrib_matrix[0][mo,ao] * \
                            cross_secs[cleanShell(data.aonames[ao])][photon]
            if len(new) == 2:
                new[1][mo,ao] = contrib_matrix[1][mo,ao] * \
                                cross_secs[cleanShell(data.aonames[ao])][photon]
    """

    return new


def writeMOFile(filename, data):
    MO_file = open(filename, 'w')
    if len(data.moenergies) == 1:
        MO_file.write('MO,(eV)\n')
    else:
        MO_file.write('MO,alpha(eV),beta(eV)')
    for i, _ in enumerate(data.moenergies[0]):
        if len(data.moenergies) == 1:
            MO_file.write(
                str(i+1) + ',%.4f' % data.moenergies[0][i] + '\n'
                )
        else:
            MO_file.write(
                str(i+1) + ',%.4f,%.4f' % (data.moenergies[0][i],data.moenergies[1][i]) + '\n'
                )
    MO_file.close()


def writeDOSFile(PDOS, energies, filename, labels = None):
    """
    write DOS spectra to file

    PDOS will be a list of np.arrays

    writing single spectrum to file will be np.array
    """
    DOSfile = open(filename, 'w')

    if not type(PDOS) == list:    
        DOSfile.write(
            'Energy,DOS\n'
            )
        for i, _ in enumerate(energies):
            DOSfile.write(
                '%.1f,%.4f\n' % (energies[i], PDOS[i])
                )

    else:
        title_line = 'Energy'

        if labels is None:
            for i in len(PDOS):
                title_line += ',Group%d' % i+1
        else:
            for x in labels:
                title_line += ',%s' % x

        title_line += '\n'

        DOSfile.write(title_line)

        for i, _ in enumerate(energies):
            line = '%.1f' % energies[i]
            
            for j, _ in enumerate(PDOS):
                line += ',%.4f' % PDOS[j][i]
            line += '\n'
            
            DOSfile.write(line)

    DOSfile.close()


# ========================
# ----- Main Program -----
# ========================

if __name__ == "__main__":

    help_message = """

Instructions:

pdos.py takes 4 positional arguments:
    1) path to output file 
    2) initial energy (eV) to compute DOS spectrum
    3) final energy (eV) 
    4) fwhm (eV) for gaussian broadening
    5) (optional) ad-hoc fwhm scaling factor:
        fwhm(E) = fwhm + c*(E-E_HOMO)

Specify groups in optional groups.txt in same directory as output file
All atoms must be present exactly once in groups.txt 

Example (20 atoms total):

atoms 
group1
1-10,11,12,15-19 
group2 
13,14,20 

    """

    if len(sys.argv) <= 4:
        print(help_message)
        sys.exit()

    Ei = int(sys.argv[2]) # plot DOS in range (Ei,Ef)
    Ef = int(sys.argv[3])
    fwhm = float(sys.argv[4])
    if len(sys.argv) == 6:
        fwhm_coeff = float(sys.argv[5]) 

    energies = [x / 10 for x in range(Ei*10, Ef*10+1)]  

    groups_filename = "groups.txt"  

    make_plots = True

    # -----------------
    # Parse Output File
    # -----------------

    output_file = os.path.basename(sys.argv[1])
    t = os.path.dirname(sys.argv[1]) # get directory part of file name
    if t:
        os.chdir(t) # if we are not already in that directoy, cd into it    

    print("Using", output_file)
    data = cclib.io.ccread(output_file)

    # ---------------------
    # Total DOS and MO file
    # ---------------------

    writeMOFile('MOs.csv', data)    

    DOS = sum_gaussians(data.moenergies[0], energies, fwhm)
    writeDOSFile(DOS, energies, 'totalDOS.csv') 

    if make_plots:
        plt.figure(1)
        plt.plot(energies, DOS)
        plt.title('Total DOS')  
    

    # check for overlap matrix
    # if missing, just compute total DOS and stop   

    if not hasattr(data, 'aooverlaps'):
        print('No overlap matrix found in output file, only computing total DOS')
        plt.show()
        sys.exit()

    # ----------------------
    # Do population analysis
    # ----------------------    

    groups = makeGroups(data, filename = groups_filename)   

    labels = list(groups.keys())
    labels.append('Total')  

    poptypes = ['MPA', 'LPA']
    photons = ['XPS', 'He2', 'He1'] 

    fignum = 2

    for ptype in poptypes:
        
        M = popAnalysis(data, poptype = ptype)  
        weights = sumOverGroups(M, groups)
        PDOS = weighted_sum_gaussians(data.moenergies[0], energies, fwhm, weights[0])
        
        fname = 'PDOS_'+ptype+'.csv'
        writeDOSFile(PDOS, energies, fname, labels=labels)  

        if make_plots:
            plt.figure(fignum)
            for i, _ in enumerate(PDOS):
                plt.plot(energies, PDOS[i], label=labels[i]) 
            plt.title(fname)
            plt.legend()
            fignum += 1 

        for g in photons:   

            M_g = applyCrossSections(M, data, cross_secs, g)
            weights_g = sumOverGroups(M_g, groups)
            PDOS_g = weighted_sum_gaussians(data.moenergies[0], energies, fwhm, weights_g[0])
            
            fname = 'PDOS_'+ptype+'_'+g+'.csv'
            writeDOSFile(PDOS_g, energies, fname , labels=labels)   

            if make_plots:
                plt.figure(fignum)
                for i, _ in enumerate(PDOS_g):
                    plt.plot(energies, PDOS_g[i], label = labels[i])
                plt.legend()
                plt.title(fname)
                fignum += 1 

    if make_plots:
        plt.show()