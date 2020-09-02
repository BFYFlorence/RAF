import matplotlib.pyplot as plt
import numpy as np

target_file = "/Users/erik/Desktop/NoIptg_ProNoBindDna/100ns_CA_tpr.xvg"
indicator = ['potential','temperature','pressure','density', 'RMSD-CA'][4]
#                 0             1           2         3         4
x = []
y = []

with open(target_file) as f:
    for i in f.readlines():
        if i[0]!="#" and i[0]!="@":
            li = i.split()
            x.append(float(li[0]))
            y.append(float(li[1]))

if len(x)==len(y):
    data = np.empty((2,len(x)),dtype=float,order='C')
    data[0] = x
    data[1] = y

fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
ax1 = fig.add_subplot(1,1,1)
ax1.set_title('Figure')
ax1.set_xlabel('ps')
ax1.set_ylabel(indicator)
plot1=ax1.plot(x,y,color='g')
plt.show()

