import matplotlib.pyplot as plt
import numpy as np

target_file = "/Users/erik/Desktop/work/cut_DNABindDomain/NoIptg_ProBindDna/Bind_rmsf.xvg"
indicator = ['potential(kJ/mol)','temperature','pressure','density', 'RMSD-CA', 'RMSF(nm)'][5]
#                 0                     1           2         3         4          5
x = []
y = []

with open(target_file) as f:
    for i in f.readlines():
        i = i.strip()
        if len(i) == 0:  # 遇见空行，表示迭代至文件末尾，跳出循环
            break
        if i[0]!="#" and i[0]!="@":
            li = i.split()
            x.append(float(li[0]))
            y.append(float(li[1]))

if len(x)==len(y):
    data = np.empty((2,len(x)),dtype=float,order='C')
    data[0] = x
    data[1] = y

print(len(y))
fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
ax1 = fig.add_subplot(1,2,1)
ax1.set_title('Figure',fontsize=20)
ax1.set_xlabel('A(aa)',fontsize=20)
ax1.set_ylabel(indicator,fontsize=20)
ax1.plot(range(62,(62+int(len(y)/2))),y[:int(len(y)/2)],color='g')

ax2 = fig.add_subplot(1,2,2)
ax2.set_title('Figure',fontsize=20)
ax2.set_xlabel('B(aa)',fontsize=20)
ax2.set_ylabel(indicator,fontsize=20)
ax2.plot(range(62,(62+int(len(y)/2))),y[int(len(y)/2):],color='g')
plt.show()

