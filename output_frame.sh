point=1
point_max=100
while [ ${point} -le ${point_max} ]
do
echo 0 | gmx trjconv -s npt.tpr -f npt.trr -o ${point}ps.pdb -dump ${point} -tu ps
point=$((${point}+1))
done
