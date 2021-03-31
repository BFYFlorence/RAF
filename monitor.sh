#!/bin/bash

# ls ./frame | while read line
# do
# stride ./frame/${line} >>./SASA/sasa_${line}
# done
md=md2
serial=part0003

mkdir frame_pbc

echo 18 1 | gmx trjconv -s ${md}.tpr -f ${md}.${serial}.trr -o ./frame_pbc/md.pdb -sep -skip 1 -pbc mol -center -n center.ndx

~/miniconda3/envs/tf/bin/python3 DisDih.py

name=md_pbc_center_nw
echo 18 1 | gmx trjconv -s ${md}.tpr -f ${md}.${serial}.trr -o ${name}.trr -skip 1 -b 1 -pbc mol -center -n center.ndx

echo 4 4 | gmx rms -s ${md}.tpr -f ${name}.trr -o rmsd.xvg -mw no

echo 4 4 | gmx rms -s ${md}.tpr -f ${name}.trr -o rmsd_weight.xvg -mw yes

