import matplotlib.pyplot as plt
import numpy as np

target_file = "/Users/erik/Desktop/metadynamics/learn/"

def hills(target_file,dictator=None):
    y = []
    with open(target_file) as f:
        for i in f.readlines():
            record = i.strip().split()
            print(record)
            if record[0][0]!='#':
                y.append(float(record[1]))

    fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title('Figure')
    ax1.set_xlabel('ps')
    ax1.set_ylabel(dictator)
    ax1.plot(range(len(y)),y,color='g',marker='+')
    plt.show()

def colvar(target_fil,dictator=None):
    y1 = []
    y2 = []
    with open(target_file) as f:
        for i in f.readlines():
            record = i.strip().split()
            print(record)
            if record[0][0] != '#':
                y1.append(float(record[1]))
                y2.append(float(record[2]))

    fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('Figure')
    ax1.set_xlabel('ps')
    ax1.set_ylabel(dictator)
    ax1.scatter(range(len(y1)), y1, color='r', marker='+')
    # ax1.scatter(range(len(y2)), y2, color='g', marker='+')
    plt.show()

def free_energy(target_file, dictator=None):
    x = []
    y = []
    with open(target_file) as f:
        for i in f.readlines():
            record = i.strip().split()
            print(record)
            if record[0][0] != '#':
                x.append(float(record[0]))
                y.append(float(record[1]))

    fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('Figure')
    ax1.set_xlabel('ps')
    ax1.set_ylabel(dictator)
    ax1.plot(x, y, color='r', marker='+')
    # ax1.scatter(range(len(y2)), y2, color='g', marker='+')
    plt.show()

def convergence(target_file, dictator=None):
    for i in range(11):
        path = target_file+"fes_{0}.dat".format(i*10)
        x = []
        y = []
        with open(path) as f:
            for i in f.readlines():
                record = i.strip().split()
                # print(record)
                if record[0][0] != '#':
                    x.append(float(record[0]))
                    y.append(float(record[1]))

        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Figure')
        ax1.set_xlabel('psi')
        ax1.set_ylabel(dictator)
        ax1.plot(x, y)
    # ax1.scatter(range(len(y2)), y2, color='g', marker='+')
    plt.show()

# hills(target_file, dictator="freeenergy")
# colvar(target_file)
# free_energy(target_file, dictator="free_energy")
convergence(target_file, dictator="free_energy")
