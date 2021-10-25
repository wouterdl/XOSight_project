import numpy as np 
import os
np_load_old = np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

bbox_data = np.load('0.npy')

np.load = np_load_old

print(bbox_data)