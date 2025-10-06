import numpy as np
import pandas as pd
import HFSCF.integrals as integrals
import math


conf = [15, 27, 30, 75, 78, 90, 39, 51, 54, 99, 102, 114, 45, 57, 60, 105, 108, 120, 135, 147, 150, 195, 198, 210, 141, 153, 156, 201, 204, 216, 165, 177, 180, 225, 228, 240]

for i in conf:
    print(i, '=', bin(i))