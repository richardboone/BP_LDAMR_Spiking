#gen_matrix.py
import numpy as np
import torch
import functions.loss_f as loss_f
import argparse
from network_parser import parse

# conv_1
# [1, 28, 28]
# [10, 12, 12]
# [10, 1, 5, 5, 1]
# -----------------------------------------
# pooling_1
# [10, 12, 12]
# [10, 6, 6]
# [1, 1, 2, 2, 1]
# -----------------------------------------
# linear
# FC_1
# [10, 6, 6]
# [100, 1, 1]
# [100, 360]
# -----------------------------------------
# linear
# output
# [100, 1, 1]
# [10, 1, 1]
# [10, 100]
# -----------------------------------------


dtype = np.float64
endings = ["5000.txt", "50000.txt", "720000.txt"]
file_prepend = "transition_"
datafile_loc = "logs/" + file_prepend
datafiles = []
for i in range(len(endings)):
    string = datafile_loc + endings[i]
    datafiles.append(string)
labelfile = "logs/transition_labels_logs.txt"
outputfile =  "logs/transition_output_logs.txt"
datanames = ["fc2", "fc1", "conv1"]

# temp_network_config = {}
# temp_network_config['n_steps'] = 10
# temp_network_config['n_class'] = 10
# temp_network_config['tau_s'] = 



def gen_targets(labels, network_config, rule="kernel"):
    label_shape = labels.shape
    # print(label_shape)
    n_class = int(network_config['n_class'])
    n_steps = int(network_config['n_steps'])
    targetshape = (int(label_shape[0]), n_class, 1, 1, n_steps)
    # for idx, item in enumerate(targetshape):
    #     targetshape[idx] = np.float64
    # print(targetshape)
    if rule == "kernel":
        # set target signal
        if n_steps >= 10:
            desired_spikes = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).repeat(int(n_steps/10))
        else:
            desired_spikes = torch.tensor([0, 1, 1, 1, 1]).repeat(int(n_steps/5))
        desired_spikes = desired_spikes.view(1, 1, 1, 1, n_steps)
        desired_spikes = loss_f.psp(desired_spikes, network_config).view(1, 1, 1, n_steps)
        targets = np.zeros(targetshape, dtype=dtype)
    else:
        print("bad rule")
        exit()
    for i in range(len(labels)):
        # print(labels[i])
        targets[i, int(labels[i]), ...] = desired_spikes
    return targets

def gen_errors_loss(outputs, targets, rule="kernel"):
    error = loss_f.SpikeLoss({})
    errval, loss = error.dfa_spike_kernel(outputs, targets, {})
    # if rule="kernel":?
        # error..dfa_spike_kernel(ouputs, )
    
def local_loss_kernel(outputs, targets):
    delta = outputs-targets
    error = delta
    loss = 1/2*np.sum(delta**2)
    return error, loss


def gen_single_matrix(data, outputs, labels, network_config, rule):
    targets = gen_targets(labels, network_config)

    n_steps = int(network_config['n_steps'])
    outputs = outputs.reshape(targets.shape)
    # error_func = loss_f.SpikeLoss(network_config)
    errval, loss = local_loss_kernel(outputs, targets)
    # error_loss = gen_errors_loss(outputs, targets)
    # print("nevermind")
    print(errval.shape, data.shape)
    errval = np.sum(errval, 4)
    # print("---------------------------")
    # print(errval[0])
    # print(data[0,])
    # print("---------------------------")
    data = np.sum(data.reshape(data.shape[0], -1, n_steps), axis=2)
    # print(data[0])
    # print()
    print(errval.shape, data.shape)
    x, re, ra, s = np.linalg.lstsq(errval.reshape((errval.shape[0], -1)), data, rcond=None)
    print(np.mean(x), np.var(x))
    # print(x)
    # print(re)
    # print("---------------------------")

    # x = x.reshape()
    # print(x)
    # print(s.shape)
    # exit()
    return x






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', action='store', dest='config', help='The path of config file')

    args = parser.parse_args()
    params = parse(args.config)
    network_config = params['Network']

    data_dict = {}
    f = open(labelfile, 'r')
    # labeldata = np.loadtxt(f, max_rows=1000)
    labeldata = np.loadtxt(f)
    f.close()
    
    f = open(outputfile, 'r')
    outputdata = np.loadtxt(f)
    # outputdata = np.loadtxt(f, max_rows=1000)
    f.close()

    matrices = []
    for i in range(len(datafiles)):
        f = open(datafiles[i], 'r')
        data = np.loadtxt(f)
        # data = np.loadtxt(f, max_rows=1000)
        print(data.shape)
        data_dict[datanames[i]] = data
        f.close()
        matrix = gen_single_matrix(data, outputdata, labeldata, network_config, "kernel")
        print(matrix.shape)
        matrices.append(matrix)
        # exit()
        # matrix = gen_single_matrix(data, outputdata, labeldata, network_config, "kernel")
        # print(matrix.shape)
        # matrices.append(matrix)
        # if (i==1):
        #     exit() 
    # exit()
    # for item in data_dict:
    #     data = data_dict[item]
    #     matrix = gen_single_matrix(data, outputdata, labeldata, network_config, "kernel")
    #     print(matrix.shape)
    #     matrices.append(matrix)

    # file_prepend = "postprocessing/matrix_alltr_slow_79_"
    for idx, item in enumerate(matrices):
        matrix = matrices[idx]
        f = open("postprocessing/" + file_prepend + str(idx) + ".txt", 'ab')
        np.savetxt(f, matrix)
        f.close()

    # print(data_dict.keys())

