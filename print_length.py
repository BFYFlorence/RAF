import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

sequence = "MWTVVLGLATLFVAYYIHWINKWRDSKFNGVLPPGTMGLPLIGETIQLSRPSDSLDVHPFIQKKVERYGPIFKTCLAGRPVVVSADAEFNNYIMLQEGRAVEMWYLDTLSKFFGLDTEWLKALGLIHKYIRSITLNHFGAEALRERFLPFIEASSMEALHSWSTQPSVEVKNASALMVFRTSVNKMFGEDAK"
# print(len(sequence))


def processIon(aa):
    if aa in ['ASP', 'ASH']:
        return 'ASP'
    if aa in ['HIS', 'HIE', 'HID', 'HIP']:
        return 'HIS'
    return aa


def mergefeat(path):
    set_con = np.load(path, allow_pickle=True).item()
    set_con = list(set_con)
    set_con.sort()

    new_set_con = set()
    for i in set_con:
        record1 = i[0].split('-')
        record2 = i[1].split('-')

        new_tuple = (record1[2]+'-'+processIon(record1[3])+'-'+record1[1],
                     record2[2]+'-'+processIon(record1[3])+'-'+record2[1])
        new_set_con.add(new_tuple)
    return new_set_con

path1 = "/Users/erik/Desktop/NoIptg_ProBindDna/csv/total_contact.npy"
path2 = "/Users/erik/Desktop/NoIptg_ProNoBindDna/csv/total_contact.npy"

new_set_con1 = mergefeat(path1)
new_set_con2 = mergefeat(path2)


print(len(new_set_con2))
print(len(new_set_con1))

set_con = new_set_con1 | new_set_con2

print(len(set_con))
np.save("/Users/erik/Desktop/NNread/" + "whole_cp.npy", set_con)