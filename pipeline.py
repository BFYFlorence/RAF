import os
from multiprocessing import Pool
from Lacomplex import Lacomplex
# ---  man-made  -- #
batch_name = 'iptg'

lc = Lacomplex()
frame_path  = './frame/'
csv_path    = './csv/'
ANN_path    = './ANN/'
output_path = '../analyse/'

for dir_name in [frame_path, csv_path, ANN_path, output_path]:
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

# 提取帧
# os.system('echo 1 | gmx trjconv -s md.tpr -f md.trr -o {0}/md.pdb -sep -skip 1 -pbc nojump'.format(frame_path))

# batch_calContact
batch_size = 20  # 任务数
frame_num = 5000

# ----------------丢弃
# 预先处理0号帧获取原子名称等信息
# lc.frame_path = frame_path
# lc.csv_path = csv_path
# lc.frame_name = 'md{0}.pdb'
# lc.csv_name = '{0}.csv'
# lc.Interval = 1
# a_atom_cor, b_atom_cor, a_atom_nam, b_atom_nam = lc.readHeavyAtom(lc.frame_path + lc.frame_name.format(0))
# lc.calContact(a_atom_cor, b_atom_cor, a_atom_nam=a_atom_nam, b_atom_nam=b_atom_nam, filename=0, save_dis=True)
# ----------------丢弃

"""def mpi_cal(serial):
    lc = Lacomplex()
    lc.startFrame = int((frame_num/batch_size)*(serial-1) + 1)
    lc.endFrame   = int((frame_num/batch_size)*serial     + 1)
    lc.frame_path = frame_path
    lc.csv_path   = csv_path
    lc.frame_name = 'md{0}.pdb'
    lc.csv_name   = '{0}.csv'
    lc.Interval   = 1
    lc.set_name   = serial
    lc.contact_dis = 4.5
    lc.batchFrame2Dis()

# 并行计算距离
serial_num = [(i+1) for i in range(batch_size)]
pool = Pool(processes=batch_size)
pool.map(mpi_cal, serial_num)
pool.close()
pool.join()

# mergeSet
lc.startSet = 1
lc.endSet = 20
lc.csv_path = csv_path
lc.output = output_path
lc.data_name = batch_name
lc.mergeSet()

# aa_contact
lc.csv_path = csv_path
lc.aa_contact()
"""
####################################断点

# 要有取并集的一步
lc.output = output_path
lc.mergebothContact()

# extractFeat
lc.csv_path = csv_path
lc.startFrame = 1
lc.endFrame = 5000 + 1
lc.csv_name = '{0}.csv'
lc.Interval = 1
lc.NNread = ANN_path
lc.output = output_path
lc.data_name = batch_name
lc.extractFeat()

# potential
os.system('echo 11 | gmx energy -f md.edr -o potential_kJ_mol.xvg')
# temperature
os.system('echo 15 | gmx energy -f md.edr -o temperature.xvg')
# pressure
os.system('echo 17 | gmx energy -f md.edr -o pressure.xvg')
lc.vmd_rmsd_path = './'
lc.output = output_path
lc.data_name = batch_name
lc.statistics()

# ConDifPerFra_CB
lc.NNread = ANN_path
lc.endFrame = 5000 + 1
lc.startFrame = 1
lc.csv_path = csv_path
lc.csv_name = '{0}.csv'
lc.Interval = 1
lc.output = output_path
lc.data_name = batch_name
lc.ConDifPerFra_CB()

# avedis
lc.csv_path = csv_path
lc.csv_name = '{0}.csv'
lc.data_name = batch_name
lc.avedis()

# after using VMD to produce RMSD data and transfering it to local PC
"""lc.rmsd_name = batch_name + '_rmsd.dat'
lc.vmd_rmsd_path = '/Users/erik/Desktop/work2/analyse/'
lc.rmsd()"""
