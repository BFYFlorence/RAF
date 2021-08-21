import numpy as np
from multiprocessing import Pool
from RAF import RAF
import matplotlib.pyplot as plt
# ---  man-made  -- #

pdb_path = "./frame_pbc"

batch_size = 10  # number of tasks
frames = 25000  # +1
def mpi_cal(serial):
    # dis_npy = []
    dih_npy_phi = []
    dih_npy_psi = []
    raf = RAF()
    for p in range(int((frames/batch_size)*(serial-1)), int((frames/batch_size)*serial)):
        # dis_value = lc.single_LDA_Dis(pdb_path+"/md{0}_recover.pdb".format(p), "extract_{0}".format(serial), p)
        # dih_value = lc.single_LDA_dih(pdb_path+"/md{0}_recover.pdb".format(p), p)
        # dis_npy.append(dis_value)
        dih_value_phi = raf.gather_dihedral_atom_singChain(pdb_path+"/md{0}.pdb".format(p), type="Phi")
        dih_value_psi = raf.gather_dihedral_atom_singChain(pdb_path+"/md{0}.pdb".format(p), type="Psi")
        dih_npy_phi.append(dih_value_phi)
        dih_npy_psi.append(dih_value_psi)

    # np.save("./dis_{0}.npy".format(serial), np.array((dis_npy)))
    np.save("./phi_{0}.npy".format(serial), np.array((dih_npy_phi)))
    np.save("./psi_{0}.npy".format(serial), np.array((dih_npy_psi)))

# parallel
serial_num = [(i+1) for i in range(batch_size)]
pool = Pool(processes=batch_size)
pool.map(mpi_cal, serial_num)
pool.close()
pool.join()
