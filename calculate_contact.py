import numpy as np
np.set_printoptions(suppress=True)  # 取消科学计数显示
import matplotlib.pyplot as plt

path = "AB.pdb"
length_pro = 328
A_atom = np.zeros(shape=(length_pro,3))    # 残基序号-数组的索引=2
B_atom = np.zeros(shape=(length_pro,3))

index = 0                           # 向atom中添加A,B两条链坐标,
with open(path,'r') as f:
    for i in f.readlines():
        record = i.strip().split()
        if len(record)==0:
            break
        if record[0]=="TER":
            index = 0
        if record[2]=="CA" and record[4]=="A":
            A_atom[index][0] = float(record[6])
            A_atom[index][1] = float(record[7])
            A_atom[index][2] = float(record[8])
            index+=1
        if record[2]=="CA" and record[4]=="B":
            B_atom[index][0] = float(record[6])
            B_atom[index][1] = float(record[7])
            B_atom[index][2] = float(record[8])
            index+=1

# 计算contact
dis_array = np.zeros(shape=(length_pro,length_pro))
contact_pair = []
for i in range(length_pro):
    for j in range(length_pro):
        x_2 = pow(A_atom[i][0] - B_atom[j][0], 2)
        y_2 = pow(A_atom[i][1] - B_atom[j][1], 2)
        z_2 = pow(A_atom[i][2] - B_atom[j][2], 2)
        dis = np.sqrt(x_2+y_2+z_2)
        if dis <= 10:
            contact_pair.append((i+2,j+2))
        dis_array[i][j] = dis

# print(A_atom[94])
# print(B_atom[94])
# print(dis_array)
print(len(contact_pair))




# 绘制热图
'''fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
ax1 = fig.add_subplot(1,1,1)
ax1.set_title('heatmap')
im = ax1.imshow(dis_array, cmap=plt.cm.hot_r)
plt.colorbar(im)
plt.show()'''