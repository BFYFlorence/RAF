start=1
end=20
while [ ${start} -le ${end} ]
do
echo ${start}
/Users/erik/miniconda3/envs/python37/bin/python3 /Users/erik/PycharmProjects/Lacomplex/Lacomplex.py ${start}
start=$((${start}+1))
done
