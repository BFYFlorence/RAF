start=1
end=20
while [ ${start} -le ${end} ]
do
echo ${start}
/Users/erik/miniconda3/envs/python37/bin/python3 /Users/erik/PycharmProjects/Lacomplex/Lacomplex.py ${start}
start=$((${start}+1))
done


#!/bin/sh
#
#PBS -N monitor1
#PBS -o LDA.log
#PBS -e LDA.error
#PBS -q low
#PBS -l nodes=1:ppn=1
#PBS -l walltime=240:00:00
cd /home/liuhaiyan/fancao/batch/1

~/miniconda3/envs/tf/bin/python3 ./monitor.py
