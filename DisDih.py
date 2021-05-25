import numpy as np
from multiprocessing import Pool
from Lacomplex import Lacomplex
import matplotlib.pyplot as plt
# ---  man-made  -- #
# batch_name = 'iptg'

lc = Lacomplex()
pdb_path = "./frame_pbc"

batch_size = 40  # 任务数
frames = 40000  # +1
def mpi_cal(serial):
    dis_npy = []
    dih_npy = []
    lc = Lacomplex()
    for p in range(int((frames/batch_size)*(serial-1)), int((frames/batch_size)*serial)):
        dis_value = lc.single_LDA_Dis(pdb_path+"/md{0}_recover.pdb".format(p), "extract_{0}".format(serial), p)
        dih_value = lc.single_LDA_dih(pdb_path+"/md{0}_recover.pdb".format(p), p)
        dis_npy.append(dis_value)
        dih_npy.append(dih_value)

    np.save("./dis_{0}.npy".format(serial), np.array((dis_npy)))
    np.save("./dih_{0}.npy".format(serial), np.array((dih_npy)))


# 并行
serial_num = [(i+1) for i in range(batch_size)]
pool = Pool(processes=batch_size)
pool.map(mpi_cal, serial_num)
pool.close()
pool.join()

