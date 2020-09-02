import numpy as np
import Lacomplex as Lc

np.set_printoptions(suppress=True)                          # 取消科学计数显示
np.set_printoptions(threshold=np.inf)                       # 没有省略号，显示全部数组


path = "/Users/erik/PycharmProjects/frames/100ps.pdb"           # ./frames/1ps.pdb

lc = Lc.Lacomplex()

# a_atom_cor, b_atom_cor, a_atom_nam, b_atom_nam = lc.readHeavyAtom(path)
# dis_array, contact_pair = lc.calContact(a_atom_cor, b_atom_cor, a_atom_nam, b_atom_nam, 100, save_dis=True)

# lc.batchFrame2Dis()
lc.readDisOutConFre()
"""
proc print_rmsd_through_time {{mol top}} {
    set reference [atomselect $mol "protein" frame 0]
    set compare [atomselect $mol "protein"]
    set num_steps [molinfo $mol get numframes]
    for {set frame 0} {$frame < $num_steps} {incr frame} {
        $compare frame $frame
        set trans_mat [measure fit $compare $reference]
        $compare move $trans_mat
        set rmsd [measure rmsd $compare $reference]
        puts "RMSD of $frame is $rmsd"
    }
}
"""