work_dir = "/Users/erik/Desktop/laci/mutations/C126/"

file1 = "A.pdb"
list1 = []
file2 = "B.pdb"
list2 = []
file3 = "A_muta_H.pqr"                                          # B_muta_H.pqr
list3 = []
file4 = "AB.pdb"
list4 = []
file5 = "B_muta_H.pqr"
list5 = []
len_pro = 328

# 读取氨基酸序列
def read_aa(path):
    aa = []
    with open(path, 'r') as f:
        n = 1                                                   # 从第二位氨基酸开始
        for i in f.readlines():
            record = i.strip().split()
            if record[0] == 'TER':
                n = 1
            if record[0] == 'ATOM' and float(record[5]) != n:
                aa.append(record[3])
                n += 1
    return aa

# 比较两条链序列的差异
def detect_diff(li1,li2):
    diff = []
    for i in range(len(li1)):
        if li1[i]!=li2[i]:
            diff.append((i+2,li2[i]))                           # 索引+2=残基序号
    return diff

#############################################
list1 = read_aa(work_dir + file1)
list2 = read_aa(work_dir + file2)
list4 = read_aa(work_dir + file4)

for i in range(len(list4)):
    if (list1+list2)[i]!=list4[i]:
        print("两条链有不同")

list3 = read_aa(work_dir + file3)
list5 = read_aa(work_dir + file5)

diff = detect_diff(list4,list3+list5)

for i in diff:
    print([i[0] if i[0]<328 else i[0]-328], list4[i[0] - 2])
    print([i[0] if i[0]<328 else i[0]-328], i[1])
    print("\n")