import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import layers, optimizers
from matplotlib.pyplot import MultipleLocator
import os
import sys
import csv
from Lacomplex import Lacomplex
lc = Lacomplex()

aa_contact = list(np.load('./aa_contact.npy', allow_pickle=True).item())
aa_contact.sort()
print(aa_contact)
