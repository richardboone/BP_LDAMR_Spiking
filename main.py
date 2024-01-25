import os

import torch
from network_parser import parse
from datasets import loadMNIST, loadFashionMNIST, loadNMNIST_Spiking, loadDVSGesture
import logging
import cnns
from utils import learningStats
from utils import aboutCudaDevices
from utils import EarlyStopping
import functions.loss_f as loss_f
import numpy as np
from datetime import datetime
import pycuda.driver as cuda
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils import clip_grad_value_
import global_v as glv
# from postprocessing import grad_evaluation
import profiler

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import argparse
# import qtorch


max_accuracy = 0
min_loss = 1000
grad_dict = {'bp':{},'rule':{}}
##########################################Visual Barrier #################################
def train(network, trainloader, opti, epoch, states, network_config, layers_config, err, early_stopping, opti2=None):
    global max_accuracy
    global min_loss
    global grad_dict
    grad_dict['bp'][epoch] = {}
    grad_dict['rule'][epoch] = {}
    logging.info('\nEpoch: %d', epoch)
    train_loss = 0
    correct = 0
    total = 0
    n_steps = network_config['n_steps']
    n_class = network_config['n_class']
    batch_size = network_config['batch_size']
    time = datetime.now()

    if network_config['loss'] == "kernel":
        # set target signal
        if n_steps >= 10:
            desired_spikes = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).repeat(int(n_steps/10))
        else:
            desired_spikes = torch.tensor([0, 1, 1, 1, 1]).repeat(int(n_steps/5))
        desired_spikes = desired_spikes.view(1, 1, 1, 1, n_steps).to(device)
        desired_spikes = loss_f.psp(desired_spikes, network_config).view(1, 1, 1, n_steps)
        targets = torch.zeros((batch_size, n_class, 1, 1, n_steps), dtype=dtype).to(device) 
    des_str = "Training @ epoch " + str(epoch)
    for batch_idx, (inputs, labels) in enumerate(trainloader):


        ########################################################
        start_time = datetime.now()
        if network_config["rule"] == "TSSLBP":
            if len(inputs.shape) < 5:
                inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
            # forward pass
            labels = labels.to(device)
            inputs = inputs.to(device)
            inputs.type(dtype)
            outputs = network.forward(inputs, epoch, True)

            if network_config['loss'] == "count":
                # set target signal
                desired_count = network_config['desired_count']
                undesired_count = network_config['undesired_count']

                targets = torch.ones((outputs.shape[0], outputs.shape[1], 1, 1), dtype=dtype).to(device) * undesired_count
                for i in range(len(labels)):
                    targets[i, labels[i], ...] = desired_count
                loss = err.spike_count(outputs, targets, network_config, layers_config[list(layers_config.keys())[-1]])
            elif network_config['loss'] == "kernel":
                targets.zero_()
                for i in range(len(labels)):
                    targets[i, labels[i], ...] = desired_spikes
                loss = err.spike_kernel(outputs, targets, network_config)
            elif network_config['loss'] == "softmax":
                # set target signal
                loss = err.spike_soft_max(outputs, labels)
            else:
                raise Exception('Unrecognized loss function.')
            # backward pass
            opti.zero_grad()

            loss.backward()
            clip_grad_norm_(network.get_parameters(), 1)
            opti.step()
            network.weight_clipper()

            spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
            predicted = np.argmax(spike_counts, axis=1)
            train_loss += torch.sum(loss).item()
            labels = labels.cpu().numpy()
            total += len(labels)
            correct += (predicted == labels).sum().item()
        elif network_config["rule"] in ["TSSLBP_DFAS"]:
            if len(inputs.shape) < 5:
                inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
            # forward pass
            labels = labels.to(device)
            inputs = inputs.to(device)
            inputs.type(dtype)

            outputs = network.forward(inputs, epoch, True)

            if network_config['loss'] == "count":
                # set target signal
                desired_count = network_config['desired_count']
                undesired_count = network_config['undesired_count']

                targets = torch.ones((outputs.shape[0], outputs.shape[1], 1, 1), dtype=dtype).to(device) * undesired_count
                for i in range(len(labels)):
                    targets[i, labels[i], ...] = desired_count

                errval, loss = err.dfa_spike_count(outputs, targets, network_config, layers_config[list(layers_config.keys())[-1]])
            elif network_config['loss'] == "kernel":
                targets.zero_()
                for i in range(len(labels)):
                    targets[i, labels[i], ...] = desired_spikes
                # loss = err.spike_kernel(outputs, targets, network_config)
                errval, loss = err.dfa_spike_kernel(outputs, targets, network_config)
            elif network_config['loss'] == "softmax":
                # set target signal
                loss = err.spike_soft_max(outputs, labels)
            else:
                raise Exception('Unrecognized loss function.')

            glv.dfa_feedback = errval
            if (network_config['rule'] in ["DFA_SLEEP", "DFA_LT", "DFA_FT"]):
                glv.output_spikes = outputs

                opti2.zero_grad()
            opti.zero_grad()


            loss.backward()
            clip_grad_norm_(network.get_parameters(), 1)
            opti.step()

            network.weight_clipper()
            spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
            predicted = np.argmax(spike_counts, axis=1)
            train_loss += torch.sum(loss).item()
            labels = labels.cpu().numpy()
            total += len(labels)
            correct += (predicted == labels).sum().item()
        elif network_config["rule"] == "TSSLBP_KP":
            if len(inputs.shape) < 5:
                inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
            # forward pass
            labels = labels.to(device)
            inputs = inputs.to(device)
            inputs.type(dtype)



            outputs = network.forward(inputs, epoch, True)

            if network_config['loss'] == "count":
                # set target signal
                desired_count = network_config['desired_count']
                undesired_count = network_config['undesired_count']

                targets = torch.ones((outputs.shape[0], outputs.shape[1], 1, 1), dtype=dtype).to(device) * undesired_count
                for i in range(len(labels)):
                    targets[i, labels[i], ...] = desired_count
                errval, loss = err.dfa_spike_count(outputs, targets, network_config, layers_config[list(layers_config.keys())[-1]])
            elif network_config['loss'] == "kernel":
                targets.zero_()
                for i in range(len(labels)):
                    targets[i, labels[i], ...] = desired_spikes
                errval, loss = err.dfa_spike_kernel(outputs, targets, network_config)
            elif network_config['loss'] == "softmax":
                # set target signal
                loss = err.spike_soft_max(outputs, labels)
            else:
                raise Exception('Unrecognized loss function.')
            # backward pass
            opti.zero_grad()


            loss.backward()

            network.kp_decay()
            opti.step()

            spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
            predicted = np.argmax(spike_counts, axis=1)
            train_loss += torch.sum(loss).item()
            labels = labels.cpu().numpy()
            total += len(labels)
            correct += (predicted == labels).sum().item()


        elif network_config["rule"] == "TPA":
            if len(inputs.shape) < 5:
                inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
            # forward pass
            labels = labels.to(device)
            inputs = inputs.to(device)
            inputs.type(dtype)
            outputs = network.forward(inputs, epoch, True)
            # exit()
            # print("time cost forward:")
            # print(round((datetime.now() - start_time).total_seconds(), 2))

            if network_config['loss'] == "count":
                # set target signal
                desired_count = network_config['desired_count']
                undesired_count = network_config['undesired_count']

                targets = torch.ones((outputs.shape[0], outputs.shape[1], 1, 1), dtype=dtype).to(device) * undesired_count
                for i in range(len(labels)):
                    targets[i, labels[i], ...] = desired_count
                loss = err.spike_count(outputs, targets, network_config, layers_config[list(layers_config.keys())[-1]])
            elif network_config['loss'] == "kernel":
                targets.zero_()
                for i in range(len(labels)):
                    targets[i, labels[i], ...] = desired_spikes
                loss = err.spike_kernel(outputs, targets, network_config)
            elif network_config['loss'] == "softmax":
                # set target signal
                loss = err.spike_soft_max(outputs, labels)
            else:
                raise Exception('Unrecognized loss function.')
            # backward pass
            opti.zero_grad()
            opti2.zero_grad()

            loss.backward()
            clip_grad_norm_(network.get_parameters(), 1)
            
            

            if (network_config["tpa_bw"] == False):
                network.TPA_train_alt_2()
                clip_grad_norm_(network.get_TPA_params(), 1)
                opti2.step()
                network.kp_decay()
            opti.step()
            network.weight_clipper()


            spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
            predicted = np.argmax(spike_counts, axis=1)
            train_loss += torch.sum(loss).item()
            labels = labels.cpu().numpy()
            total += len(labels)
            correct += (predicted == labels).sum().item()

            if (network_config["altset"] == "tpa_test"):
                weight_distance = err.spike_kernel(network.layers[-1].weight, network.layers[-1].backward_matrix, network_config)
                if (states.alt.minloss == None):
                    states.alt.minloss = weight_distance
                    states.alt.lossSum += weight_distance
                    states.alt.numSamples +=1
                else:
                    if (states.alt.minloss > weight_distance):
                        states.alt.minloss = weight_distance
                    states.alt.lossSum += weight_distance
                    states.alt.numSamples +=1


        else:
            raise Exception('Unrecognized rule name.')

        states.training.correctSamples = correct
        states.training.numSamples = total
        states.training.lossSum += loss.cpu().data.item() 
        states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())

    total_accuracy = correct / total
    total_loss = train_loss / total
    if total_accuracy > max_accuracy:
        max_accuracy = total_accuracy
    if min_loss > total_loss:
        min_loss = total_loss

    logging.info("Train Accuracy: %.3f (%.3f). Loss: %.3f (%.3f)\n", 100. * total_accuracy, 100 * max_accuracy, total_loss, min_loss)

##########################################Visual Barrier #################################
def test(network, testloader, epoch, states, network_config, layers_config, early_stopping, runES=True):
    global best_acc
    global best_epoch
    correct = 0
    total = 0
    n_steps = network_config['n_steps']
    n_class = network_config['n_class']
    time = datetime.now()
    y_pred = []
    y_true = []
    des_str = "Testing @ epoch " + str(epoch)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            if network_config["rule"] == "TSSLBP":
                if len(inputs.shape) < 5:
                    inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
                # forward pass
                labels = labels.to(device)
                inputs = inputs.to(device)
                outputs = network.forward(inputs, epoch, False)

                spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
                predicted = np.argmax(spike_counts, axis=1)
                labels = labels.cpu().numpy()
                y_pred.append(predicted)
                y_true.append(labels)
                total += len(labels)
                correct += (predicted == labels).sum().item()
            elif network_config["rule"] in ["TSSLBP_DFAS"]:
                if len(inputs.shape) < 5:
                    inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
                # forward pass
                labels = labels.to(device)
                inputs = inputs.to(device)
                outputs = network.forward(inputs, epoch, False)

                spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
                predicted = np.argmax(spike_counts, axis=1)
                labels = labels.cpu().numpy()
                y_pred.append(predicted)
                y_true.append(labels)
                total += len(labels)
                correct += (predicted == labels).sum().item()
            elif network_config["rule"] == "TSSLBP_KP" or network_config["rule"] == "TPA":
                if len(inputs.shape) < 5:
                    inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
                # forward pass
                labels = labels.to(device)
                inputs = inputs.to(device)
                outputs = network.forward(inputs, epoch, False)

                spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
                predicted = np.argmax(spike_counts, axis=1)
                labels = labels.cpu().numpy()
                y_pred.append(predicted)
                y_true.append(labels)
                total += len(labels)
                correct += (predicted == labels).sum().item()    
            else:
                raise Exception('Unrecognized rule name.')

            states.testing.correctSamples += (predicted == labels).sum().item()
            states.testing.numSamples = total
            states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())

    test_accuracy = correct / total
    if test_accuracy > best_acc:
        best_acc = test_accuracy
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        cf = confusion_matrix(y_true, y_pred, labels=np.arange(n_class))
        df_cm = pd.DataFrame(cf, index = [str(ind*25) for ind in range(n_class)], columns=[str(ind*25) for ind in range(n_class)])
        plt.figure()
        sn.heatmap(df_cm, annot=True)
        plt.savefig("confusion_matrix.png")
        plt.close()

    logging.info("Train Accuracy: %.3f (%.3f).\n", 100. * test_accuracy, 100 * best_acc)
    # Save checkpoint.
    acc = 100. * correct / total
    if(runES):
        early_stopping(acc, network, epoch)
    if (network_config["saveall"]):        
        filestring = "testing_quantization_" + network_config["dataset"] + "_" + network_config["rule"] + "_lr" + str(network_config["lr"]) + "ts" + str(network_config["tau_s"]) + "_acc" + str(acc)
        early_stopping.save_checkpoint(network, acc, epoch, filestring + "_checkpoint.pth")

##########################################Visual Barrier #################################
if __name__ == '__main__':
    print("Running")
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', action='store', dest='config', help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint', help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-profile', action='store', dest='profile', default = '', help='profile config file location.  Blank if no profiling')
    parser.add_argument('-extract', action='store', dest='extract', default = False, help='extract backprop info from checkpoint, requires checkpoint')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.config is None:
        raise Exception('Unrecognized config file.')
    else:
        config_path = args.config


    
    params = parse(config_path)
    
    
    dtype = torch.float32
    
    extension = ".log"
    logging_filename = params["Network"]["dataset"] + '_r' + params["Network"]["rule"] + '_e' + str(params["Network"]["epochs"]) +  '_b' + str(params["Network"]["batch_size"]) + '_lr' + str(params["Network"]["lr"])
    if (params["Network"]["rule"] == "TSSLBP_DFA"):
        logging_filename += "_" + params["Network"]["dfa_genset"]
    logging.basicConfig(filename= logging_filename + extension, level=logging.INFO)

    # Check whether a GPU is available
    if torch.cuda.is_available() and True:
        device = torch.device("cuda:1")
        cuda.init()
        c_device = aboutCudaDevices()
        print(c_device.info())
        print("selected device: ", device)
    else:
        device = torch.device("cpu")
        print("No GPU is found")
    
    try:
        glv.init(dtype, device, params['Network']['n_steps'], params['Network']['tau_s'], lt=params["Network"]["limit_time"], network_config = params["Network"] )
    except:
        print("no lt present")
        print(params['Network']['n_steps'])
        glv.init(dtype, device, params['Network']['n_steps'], params['Network']['tau_s'], network_config = params["Network"] )

    if args.profile != '':
        print("profiling code")
        glv.profile = True

    logging.info("dataset loaded")
    if params['Network']['dataset'] == "MNIST":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadMNIST.get_mnist(data_path, params['Network'])
    elif params['Network']['dataset'] == "NMNIST_Spiking":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadNMNIST_Spiking.get_nmnist(data_path, params['Network'])
    elif params['Network']['dataset'] == "FashionMNIST":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadFashionMNIST.get_fashionmnist(data_path, params['Network'])
    elif params['Network']['dataset'] == "CIFAR10":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadCIFAR10.get_cifar10(data_path, params['Network'])
    elif params['Network']['dataset'] == "DVS_Gesture":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadDVSGesture.get_dvs_gesture(data_path, params['Network'])
    else:
        raise Exception('Unrecognized dataset name.')
    logging.info("dataset loaded")
    
    net = cnns.Network(params['Network'], params['Layers'], list(train_loader.dataset[0][0].shape), device).to(device)
    
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        # net.load_state_dict(checkpoint)
        net.load_state_dict(checkpoint['net'])
    
    error = loss_f.SpikeLoss(params['Network']).to(device)
    
    net.add_err(error)

    optimizer = torch.optim.AdamW(net.get_parameters(), lr=params['Network']['lr'], betas=(0.9, 0.999))

    if params["Network"]["rule"] == "TPA":
        optimizer2 = torch.optim.AdamW(net.get_TPA_params(), lr=params['Network']['back_lr'], betas=(0.9, 0.999))


    print(net.get_parameters())
    print(params['Network'])

    best_acc = 0
    best_epoch = 0
    
    l_states = learningStats(gradients=True)
    early_stopping = EarlyStopping()



    for e in range(params['Network']['epochs']):
        l_states.training.reset()
        # try:
        if (params["Network"]["rule"] in ["TPA"]):
            train(net, train_loader, optimizer, e, l_states, params['Network'], params['Layers'], error, early_stopping, optimizer2)
        else:
            train(net, train_loader, optimizer, e, l_states, params['Network'], params['Layers'], error, early_stopping)
        l_states.training.update()
        l_states.testing.reset()
        test(net, test_loader, e, l_states, params['Network'], params['Layers'], early_stopping)
        l_states.testing.update()
        if early_stopping.early_stop:
            break
    if params['Network']["rule"] == "TSSLBP_KP":
        if (params['Network']['kp_check'] == True):
            net.kp_matrix_check(filename='logs/test_')

    logging.info("Best Accuracy: %.3f, at epoch: %d \n", best_acc, best_epoch)
