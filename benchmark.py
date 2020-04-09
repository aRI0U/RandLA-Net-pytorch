import time
import numpy as np
import os
from utils.ply import read_ply
import sys

for path1 in sys.argv[1:]:
    path2 = path1[:-3] + 'npy'
    if os.path.exists(path2):
        continue
    print(path1)
    t0 = time.time()
    arr = read_ply(path1)
    pts = np.vstack((arr['x'], arr['y'], arr['z'], arr['intensity'], arr['red'], arr['green'], arr['blue'], arr['class'])).T
    t1 = time.time()
    print(pts.shape, pts.dtype, t1-t0)
    np.save(path2, pts.astype(np.float32), allow_pickle=False)

    t0 = time.time()
    arr = np.load(path2, mmap_mode='r')
    t1 = time.time()
    print(arr.shape, arr.dtype, t1-t0)
