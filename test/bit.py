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

def set_bit(value, bit):
    return value | (1 << bit)

nelec=2
nso = 4


a = 0       # 二進位：0000 0000
'''
#for i in range
a = set_bit(a, 10)
a = set_bit(a, 2)
a= 253


print(bin(a))
print(bin(a).count('1'))    # 輸出：4
'''

def find_occ(config,nso):
    list=[]
    for i in range(nso):
        if config & (1<<i):
            list.append(i)
    return list


config=[]
for i in range(int(2**(nso))):
    if bin(i).count('1') == nelec :
        config.append(i)

for j in range(len(config)):
    print(bin(config[j]))


print("XXXXXXXXXXXXXXX")

print(bin(config[0]))

list = find_occ(config[0],nso)

print(list)


# i in range(nso):
#    if config[5]&(1<<i):
#       print(i)

