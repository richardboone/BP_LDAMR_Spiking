import torch
import numpy as np
import network_parser
from datetime import datetime
import global_v as glv
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt



class Profile_variable():
    def __init__(self, total_size):
        self.index = 0
        print(total_size)
        total_size = [int(i) for i in total_size]
        self.logs = np.ndarray(total_size)
        self.inputshape = total_size[1:]

    def append(self, new_inputs):
        inputsize = new_inputs.shape[0]
        if (new_inputs.shape[1:] != self.logs.shape[1:]):
            raise Exception('invalid input selection, should be shape ', self.logs.shape, new_inputs.shape)
        self.logs[self.index:self.index+inputsize] = new_inputs.cpu().detach().numpy()
        self.index += inputsize

    def get_temporal_sum(self):
        return self.logs.sum(axis=4)
        


#set of profile variables, should generally only be used inside profiler
#requires that variables are inserted in the same order that the initial sizes are in
class Profile_variable_set():
    def __init__(self, num_variables, sizes):
        self.profile_variables = []
        self.index = 0
        for i in range(num_variables):
            self.profile_variables.append(Profile_variable(sizes[i]))

    def insert(self, new_inputs):
        self.profile_variables[self.index].append(new_inputs)
        self.index = (self.index + 1)% len(self.profile_variables)
    
    def get_temporal_sums(self):
        time_sums = []
        for item in self.profile_variables:
            tempsum = item.get_temporal_sum()
            time_sums.append(tempsum)
        return time_sums


class Profiler():
    def __init__(self, network, network_config, profile_config):
        self.profile_config = network_parser.parse(profile_config)
        total_samples = self.profile_config["num_samples"]
        dfa_feedback_options = ["DFA_SLEEP"]
        layer_output_shapes = []
        for layer in network.layers:
            temp_shape = list(layer.out_shape)
            temp_shape.insert(0, total_samples)
            temp_shape.append(network_config["n_steps"])
            layer_output_shapes.append(temp_shape)
        print(layer_output_shapes)
        self.layer_spikes = Profile_variable_set(len(layer_output_shapes), layer_output_shapes)
        self.layer_outputs = Profile_variable_set(len(layer_output_shapes), layer_output_shapes)
        #layer-level feedback is created in reversed order because layers values will be inserted as network is backpropagated
        self.layer_feedback = Profile_variable_set(len(layer_output_shapes), layer_output_shapes[::-1])
        if network_config["rule"] in dfa_feedback_options:
            self.layer_dfa_feedback = Profile_variable_set(len(layer_output_shapes), layer_output_shapes[::-1])

        output_size = network.layers[-1].out_shape
        output_size.insert(0, total_samples)
        self.output_spikes = Profile_variable(output_size)
        self.output_errors = Profile_variable(output_size)
        self.total_loss = Profile_variable((total_samples/network_config["batch_size"], 1))

    def insert_layer_spikes(self, new_spike):
        self.layer_spikes.insert(new_spike)

    def insert_layer_outputs(self, new_outputs):
        self.layer_outputs.insert(new_outputs)


    def insert_layer_feedback(self, layer_feedback):
        self.layer_feedback.insert(layer_feedback)

    def insert_dfa_feedback(self, dfa_feedback):
        self.layer_dfa_feedback.insert(dfa_feedback)

    def num_spikes_histogram(self, filename = ''):


        temporal_sums = self.layer_outputs.get_temporal_sums()
        # temporal_sums = self.layer_spikes.get_temporal_sums()
        # print(temporal_sums.shape)
        for i, item in enumerate(temporal_sums):
            temporal_sums[i] = item.flatten()

        temporal_sums_total = np.concatenate(temporal_sums)
        print(temporal_sums_total.shape)
        print(np.sum(temporal_sums_total))
        print(np.sum(temporal_sums_total)/temporal_sums_total.shape[0])
        
        plot_1 = plt.hist(temporal_sums_total, 50)

        # plt.savefig(filename)

        date_prepend = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        if filename == '':
            out_filename = 'postprocessing/' + date_prepend + "_histogram_num_spikes_" + "general.png"
        else:
            out_filename = filename
        plt.savefig('postprocessing/graphs/temp_test_hist.png')
        plt.savefig(out_filename)

        # print(temporal_sums)
        # for array in temporal_sums:   
        #     print(array.shape)

    def post_process(self):
        self.num_spikes_histogram()


profiler = None



def init_profiler(network, network_config, profile_config):
    global profiler
    profiler = Profiler(network, network_config, profile_config)


def profile(network, trainloader, testloader, opti, network_config, layers_config, err,  opti2=None):
    device = glv.device
    dtype = glv.dtype
    n_steps = network_config['n_steps']
    n_class = network_config['n_class']
    batch_size = network_config['batch_size']

    ###########
    total_loss = 0
    correct = 0
    total = 0

    if network_config['loss'] == 'kernel':
        # set target signal
        if n_steps >= 10:
            desired_spikes = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).repeat(int(n_steps/10))
        else:
            desired_spikes = torch.tensor([0, 1, 1, 1, 1]).repeat(int(n_steps/5))
        desired_spikes = desired_spikes.view(1, 1, 1, 1, n_steps).to(device)
        desired_spikes = loss_f.psp(desired_spikes, network_config).view(1, 1, 1, n_steps)
        targets = torch.zeros((batch_size, n_class, 1, 1, n_steps), dtype=dtype).to(device) 

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        start_time = datetime.now()
        if network_config["rule"] == "TSSLBP" or network_config["rule"] == "TSSLBP_NI":
            if len(inputs.shape) < 5:
                inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
            # forward pass
            labels = labels.to(device)
            inputs = inputs.to(device)
            inputs.type(dtype)
            outputs = network.forward(inputs, 0, True)
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

            loss.backward()
            clip_grad_norm_(network.get_parameters(), 1)
            # opti.step()
            network.weight_clipper()

            spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
            predicted = np.argmax(spike_counts, axis=1)
            total_loss += torch.sum(loss).item()
            labels = labels.cpu().numpy()
            total += len(labels)
            correct += (predicted == labels).sum().item()