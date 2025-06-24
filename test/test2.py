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

#import HFSCF.HF as HF

import sys
from scipy import constants
from pyscf.tools import molden
from pyscf.mcscf import dmet_cas
import csv

print("hellow")
count = 0
while (count < 9):
   print (count)
   count = count + 1
 
print ("Good bye!")
A = np.array([[3,6],
                [7,15]])

print(np.sum(A))