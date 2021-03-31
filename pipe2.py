import os
from multiprocessing import Pool
from Lacomplex import Lacomplex
# ---  man-made  -- #
batch_name = 'NoIptg_NoBind'

lc = Lacomplex()
# frame_path  = './frame/'
csv_path    = './csv_new/'
# ANN_path    = './ANN/'
output_path = '../analyse_new/'


for dir_name in [csv_path, output_path]:
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

# batch_calContact
batch_size = 20  # 任务数
frame_num = 5000

# 要有取并集的一步
lc.output = output_path
lc.mergebothContact()

"""# extractFeat
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
"""
# ConDifPerFra_CB
lc.endFrame = 5000 + 1
lc.startFrame = 1
lc.csv_path = csv_path
lc.csv_name = '{0}.csv'
lc.Interval = 1
lc.output = output_path
lc.data_name = batch_name
lc.ConDifPerFra_CB()

"""# avedis
lc.csv_path = csv_path
lc.csv_name = '{0}.csv'
lc.data_name = batch_name
lc.avedis()
"""