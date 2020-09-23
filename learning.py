import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import re
from tensorflow import keras
import tensorflow as tf
from matplotlib.pyplot import MultipleLocator

np.set_printoptions(threshold=np.inf)


# print(np.load('./train.npy', allow_pickle=True).shape)

cp = list(np.load('./aa_contact.npy', allow_pickle=True).item())
cp.sort()
print(cp)

aa = set()
for i in cp:
    aa.add(i[0])
    aa.add(i[1])

print(list(aa))
print(len(list(aa)))

# 2
map = np.zeros(shape=(328,328))
for i in cp:
    map[int(i[1].split('-')[0])][int(i[0].split('-')[0])] += 1

fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
ax1 = fig.add_subplot(1, 1, 1)



x_major_locator=MultipleLocator(20)
#把x轴的刻度间隔设置为50，并存在变量里
y_major_locator=MultipleLocator(20)
#把y轴的刻度间隔设置为50，并存在变量里
ax1.xaxis.set_major_locator(x_major_locator)
ax1.yaxis.set_major_locator(y_major_locator)

plt.xlim(2,329)
plt.ylim(2,329)



ax1.set_title('Figure', fontsize=20)
ax1.set_xlabel('A', fontsize=20)
ax1.set_ylabel("B", fontsize=20)
ax1.imshow(map, cmap="viridis")


# plt.show()