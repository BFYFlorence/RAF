import os
from multiprocessing import Pool
from Lacomplex import Lacomplex
# ---  man-made  -- #


lc = Lacomplex()
lc.csv_path = '/Users/erik/Desktop/work2/analyse/'
lc.output = '/Users/erik/Desktop/work2/analyse/'

# ---- both ---- #
# covariance
# lc.covariance()

# aveDistribution
lc.csv_name = '{0}.csv'
lc.aveDistribution()


# LDA
# lc.LDA()