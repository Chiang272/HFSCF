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

mol.atom =[ [ 'H',  (0, 0, 0)],
            [ 'H',  (0, 0, 1.1)]]

#mol.basis = bse.get_basis('ano-rcc',uncontract_general=True, uncontract_segmented=True, uncontract_spdf=True,elements=['Hg','H'],fmt='nwchem',header=False)
mol.basis = "3-21g"
#mol.basis = "sto-3g"


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

H_FCI = HF.FullCI(mf)


cisolver = pyscf.fci.FCI(mf)
#cisolver.analyze()
print('E(FCI) = %.12f' % cisolver.kernel()[0])


#print("PYSCF FCI")
#print(cisolver.kernel()[1])

mc = pyscf.mcscf.CASCI(mf, 4, 2)
mc.fcisolver.nroots = 16 
#emc = mc.mc1step()[0]
mc.kernel()
print( mc.ci[0])
#exit()
evec = np.zeros([16,16])
#print(mc.e_cas)

E = np.diag(mc.e_cas)
#print(E)
#print(mc.ci[0].reshape([1,4]))

for i in range(16):
    evec[:,i] = mc.ci[i].reshape([1,16])

#print(evec)

H_cas = evec @ E @ evec.T
print(H_FCI)
print("##############################")
print(H_cas)

H_FCI = np.round(H_FCI,8)
H_cas = np.round(H_cas,8)

print("WSWWWW")
print(H_FCI - H_cas)
H_dif = H_FCI - H_cas


print(np.diag(H_FCI))
print(np.diag(H_cas))
print(np.diag(H_FCI) - np.diag(H_cas))

with open('H_FCI_3-21g.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(H_FCI.tolist())

csvfile.close()


with open('H_cas_3-21g.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(H_cas.tolist())

csvfile.close()

with open('H_dif_3-21g.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(H_dif.tolist())

csvfile.close()
