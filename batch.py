import os

"""Current working dir is ~/batch"""
start_point = "200"

batch_size = 1
ppn = 4
quene = "gpu"

serial_num = [i for i in range(1, batch_size+1)]
conf = '/home/liuhaiyan/fancao/test_output'
cwd = os.getcwd()

# str = ["0123", "4567"]

for j in serial_num:
    os.system("mkdir {0}/{1}".format(cwd, j))
    os.system("cp {0}/md.tpr {0}/{1}".format(cwd,j))
    os.system("cp {0}/em.pbs {0}/{1}".format(cwd,j))
    os.system("cp {0}/em.pbs {0}/{1}/{2}.pbs".format(cwd, j, "LDA"))
    os.system("cp {0}/* {1}/{2}".format(conf, cwd, j))

    # editing em.pbs
    os.system('echo "#PBS -N {2}_MD_{0}_{3}" >> {1}/{0}/em.pbs'.format(j, cwd, quene, start_point))
    os.system('echo "#PBS -q {0}" >> {1}/{2}/em.pbs'.format(quene, cwd, j))
    os.system('echo "#PBS -l nodes={0}:ppn={1}" >> {2}/{3}/em.pbs'.format(1, ppn, cwd, j))
    os.system('echo "#PBS -o {0}/{1}/job.log" >> {0}/{1}/em.pbs'.format(cwd,j))
    os.system('echo "#PBS -e {0}/{1}/job.error" >> {0}/{1}/em.pbs'.format(cwd, j))
    os.system('echo "cd {0}/{1}" >> {0}/{1}/em.pbs'.format(cwd,j))
    os.system('echo "gmx mdrun -v -deffnm md" >> {0}/{1}/em.pbs'.format(cwd, j))
    # os.system("qsub {0}/{1}/em.pbs".format(cwd,j))

    # editing LDA.pbs
    os.system('echo "#PBS -N {2}_LDA_{0}_{3}" >> {1}/{0}/LDA.pbs'.format(j, cwd, quene, start_point))
    os.system('echo "#PBS -q {0}" >> {1}/{2}/LDA.pbs'.format("low", cwd, j))
    os.system('echo "#PBS -l nodes={0}:ppn={1}" >> {2}/{3}/LDA.pbs'.format(1, 1, cwd, j))
    os.system('echo "#PBS -o {0}/{1}/LDA.log" >> {0}/{1}/LDA.pbs'.format(cwd, j))
    os.system('echo "#PBS -e {0}/{1}/LDA.error" >> {0}/{1}/LDA.pbs'.format(cwd, j))
    os.system('echo "cd {0}/{1}" >> {0}/{1}/LDA.pbs'.format(cwd, j))
    os.system('echo "~/miniconda3/envs/tf/bin/python3 ./monitor.py" >> {0}/{1}/LDA.pbs'.format(cwd, j))
    # os.system("qsub {0}/{1}/LDA.pbs".format(cwd, j))
