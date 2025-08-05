import unittest
import numpy as np
import math
import time

import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.fci
import pyscf.mp

#import prism_beta.interface as interface
#import prism_beta.qd_nevpt as qd_nevpt
import basis_set_exchange as bse
#import prism_beta.integrals as integrals

import HFSCF.HF as HF

import sys
from scipy import constants
from pyscf.tools import molden
from pyscf.mcscf import dmet_cas
import csv

print("hellow")

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)


# SCF cycle:

#theta = np.pi/3
#x = 1.178 * np.cos(theta)
#y = 1.178 * np.sin(theta)

mol = pyscf.gto.Mole()
mol.verbose = 4

#r= 0.96
#x = r * math.sin(104.5 * math.pi/(2 * 180.0))
#y = r * math.cos(104.5 * math.pi/(2 * 180.0))
#mol.atom = [
#['O', ( 0., 0.    , 0.0)],
#['H', ( 0., -x, y)],
#['H', ( 0., x , y)],]


#mol.atom =[[ 'Li',  (0, 0, 0)]]

mol.atom =[ [ 'H',  (0, 0, 0)],[ 'H',  (0, 0, 1.1)] ]

#mol.basis = bse.get_basis('ano-rcc',uncontract_general=True, uncontract_segmented=True, uncontract_spdf=True,elements=['Hg','H'],fmt='nwchem',header=False)
mol.basis = 'sto-3g'  #'6-31g**' #'def2tzvp' #'cc-pvdz' #'STO-3G' # 'def2tzvp'


mol.symmetry = False
mol.spin = 0
#mol.unit = 'B' 
mol.max_memory =  10000
mol.charge = 0
mol.build()



# Run RHF computation
mf = pyscf.scf.RHF(mol) #.x2c()
#mf.xc = "bp86"
mf.conv_tol = 1e-12
#mf.chkfile = "mf4.chk"
ehf = mf.scf()
mf.analyze()

HF.FullCI(mf)


cisolver = pyscf.fci.FCI(mf)
#cisolver.analyze()
#print('E(FCI) = %.12f' % cisolver.kernel()[0])

print(cisolver.kernel()[0])

#mc = pyscf.mcscf.CASSCF(mf, 4, 2)
#emc = mc.mc1step()[0]
#mc.analyze()