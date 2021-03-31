#msmbuilder imports
from msmbuilder.dataset import dataset
from msmbuilder.featurizer import AtomPairsFeaturizer
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import ContinuousTimeMSM
from msmbuilder.utils import verbosedump,verboseload
from msmbuilder.preprocessing import RobustScaler

#other imports
import os,glob
import numpy as np
import mdtraj as md
import pandas as pd
import pickle

#prettier plots
#import seaborn as sns
np.set_printoptions(suppress=True)  # 取消科学计数显示

trajectory_folder_path="/Users/erik/PycharmProjects/Lacomplex"
ds = dataset(trajectory_folder_path+"/nw1.trr", stride = 1,
             topology =trajectory_folder_path + "/md0.pdb")
#get some stats about the dataset

print("Number of trajectories is %d"%len(ds))

#define our featurizer
atom_pair = [[0, 1489],
             [1, 1489],
             [2, 1489],
             [3, 1489],
             [4, 1489],
             [5, 1489],
             [6, 1489]]
atom_pair = np.array(atom_pair)
feat = AtomPairsFeaturizer(pair_indices=atom_pair, periodic=False)

#transform the data with the featurizer
ds_out = 'atom_pair_dis/'
# ds.fit
# ds_dis=ds.fit_transform_with(feat, ds_out, fmt='dir-npy')

#we can load the features using
ds_dis = dataset("./atom_pair_dis/")

#see how many trajectories were retained during featurization
print(ds_dis[0].shape)

scaler = RobustScaler()
# scaled_diheds = ds_dis.fit_transform_with(scaler, 'scaled_dis/', fmt='dir-npy')
scaled_dis = dataset("./scaled_dis/")
print(ds_dis[0])
print(scaled_dis[0])

tica_model = tICA(lag_time=2, n_components=4) # n_components要小于原特征数量
# fit and transform can be done in seperate steps:
tica_model = scaled_dis.fit_with(tica_model)
# tica_trajs = scaled_dis.fit_transform_with(tica_model, 'ticas/', fmt='dir-npy')

print(ds_dis[0].shape)
# print(tica_trajs[0].shape)

tica_trajs = dataset("./ticas/")

