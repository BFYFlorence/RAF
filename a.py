import numpy as np
# 数据组织形式：A_Phi, A_Psi, B_Phi, B_Psi

# ARG = []
test = [1,23,45,213,4,5,23,1545,436,124]  # 453.41315596263854

test = np.array(test)

mid = np.sort(test)[int(len(test)/2)]

diff = test-mid

print(mid)

print(np.std(diff))



