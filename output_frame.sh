target=md3078_110.pdb
cp ~/cluster_MDP/*.mdp ./

gmx pdb2gmx -f ${target} -o cluster.gro -ignh  # CHARMM27、TIP3P

gmx editconf -f cluster.gro -o cluster_newbox.gro -c -d 3.0 -bt dodecahedron

gmx solvate -cp cluster_newbox.gro -cs spc216.gro -o cluster_solv.gro -p topol.top

gmx grompp -f ions.mdp -c cluster_solv.gro -p topol.top -o ions.tpr

gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -conc 0.1 -neutral


# simulation

gmx grompp -f steep.mdp -c solv_ions.gro -p topol.top -o steep.tpr
gmx mdrun -v -deffnm steep -gpu_id 0 -ntmpi 1

gmx grompp -f cg.mdp -c steep.gro -p topol.top -o cg.tpr
gmx mdrun -v -deffnm cg -gpu_id 0 -ntmpi 1

nvt=100ps_0K
em=cg
gmx grompp -f ${nvt}.mdp -c ${em}.gro -r ${em}.gro -p topol.top -o ${nvt}.tpr
gmx mdrun -v -deffnm ${nvt} -gpu_id 0 -ntmpi 1

nvt_p=100ps_0K
nvt_c=900ps_0-300K
gmx grompp -f ${nvt_c}.mdp -c ${nvt_p}.gro -t ${nvt_p}.cpt -r ${nvt_p}.gro -p topol.top -o ${nvt_c}.tpr
gmx mdrun -v -deffnm ${nvt_c} -gpu_id 0 -ntmpi 1

nvt_p=900ps_0-300K
nvt_c=100ps_300K
gmx grompp -f ${nvt_c}.mdp -c ${nvt_p}.gro -t ${nvt_p}.cpt -r ${nvt_p}.gro -p topol.top -o ${nvt_c}.tpr
gmx mdrun -v -deffnm ${nvt_c} -gpu_id 0 -ntmpi 1

nvt=100ps_300K
gmx grompp -f npt.mdp -c ${nvt}.gro -t ${nvt}.cpt -r ${nvt}.gro -p topol.top -o npt.tpr
gmx mdrun -v -deffnm npt -gpu_id 0 -ntmpi 1

gmx grompp -f md.mdp -c npt.gro -t npt.cpt -r npt.gro -p topol.top -o md.tpr
gmx mdrun -v -deffnm md -gpu_id 0 -ntmpi 1

# you should prepare center.idx ahead, and this is for IPTG, so 24 is selected
mkdir frame
mkdir frame_pbc
mkdir GRO
mkdir nojump

name=md_nw
echo 1 | gmx trjconv -s md.tpr -f md.trr -o ${name}.trr -skip 1 -b 1

echo 18 1 | gmx trjconv -s md.tpr -f md.trr -o ./frame_pbc/md.pdb -sep -skip 1 -pbc mol -center -n center.ndx
echo 1 | gmx trjconv -s md.tpr -f md.trr -o ./frame/md.pdb -sep -skip 1
echo 24 0 | gmx trjconv -s md.tpr -f md.trr -o ./GRO/md.gro -sep -skip 1 -pbc mol -center -n center.ndx
echo 1 | gmx trjconv -s md.tpr -f md.trr -o ./nojump/mdnojump.pdb -sep -skip 1 -pbc nojump


