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
os.system('echo 1 | gmx trjconv -s md.tpr -f md.trr -o {0}/md.pdb -sep -skip 1 -pbc nojump'.format(frame_path))

# batch_calContact
batch_size = 20  # 任务数
frame_num = 5000
def mpi_cal(serial):
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
