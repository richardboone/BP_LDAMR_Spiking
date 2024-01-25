import torch
import torch.nn as nn
import layers.conv as conv
import layers.pooling as pooling
import layers.dropout as dropout
import layers.linear as linear
import functions.TSSLBP as f
import global_v as glv
import functions.loss_f as loss

network_config_rules = ["TSSLBP", "TSSLBP_DFAQ","TSSLBP_KP", "TPA"]

class Network(nn.Module):
    def __init__(self, network_config, layers_config, input_shape, device):
        super(Network, self).__init__()
        self.layers = []
        self.network_config = network_config
        self.layers_config = layers_config
        self.local_loss = loss.SpikeLoss(network_config).to(device)
        parameters = []
        # if (self.network_config["rule"] == )
        print("Network Structure:")
        for key in layers_config:
            c = layers_config[key]
            if "file" in c:
                dfafile = c["file"]
            else:
                dfafile = ""
            if c['type'] == 'conv':
                self.layers.append(conv.ConvLayer(network_config, c, key, input_shape, device, dfafile=dfafile))
                self.layers[-1].to(glv.device)
                input_shape = self.layers[-1].out_shape
                parameters.extend(self.layers[-1].get_parameters())
            elif c['type'] == 'linear':
                self.layers.append(linear.LinearLayer(network_config, c, key, input_shape, device, dfafile=dfafile))
                self.layers[-1].to(glv.device)
                input_shape = self.layers[-1].out_shape
                parameters.extend(self.layers[-1].get_parameters())
            elif c['type'] == 'pooling':
                self.layers.append(pooling.PoolLayer(network_config, c, key, input_shape))
                self.layers[-1].to(glv.device)
                input_shape = self.layers[-1].out_shape
            elif c['type'] == 'dropout':
                self.layers.append(dropout.DropoutLayer(c, key))
            else:
                raise Exception('Undefined layer type. It is: {}'.format(c['type']))
        self.my_parameters = nn.ParameterList(parameters)
        print("-----------------------------------------")

    def forward(self, spike_input, epoch, is_train, bp_pass=False):
        spikes = f.psp(spike_input, self.network_config)
        skip_spikes = {}
        assert self.network_config['model'] == "LIF"
        
        for i in range(len(self.layers)):
            if self.layers[i].type == "dropout":
                if is_train:
                    spikes = self.layers[i](spikes)
            elif self.network_config["rule"] in network_config_rules:
                spikes = self.layers[i].forward_pass(spikes, epoch, bp_pass=bp_pass)
            else:
                raise Exception('Unrecognized rule type. It is: {}'.format(self.network_config['rule']))


        return spikes

    def forward_bypass(self, spike_input, epoch, is_train, feedback=None):
        spikes = f.psp(spike_input, self.network_config)
        skip_spikes = {}
        assert self.network_config['model'] == "LIF"
        
        for i in range(len(self.layers)):
            if self.layers[i].type == "dropout":
                if is_train:
                    spikes = self.layers[i](spikes)
            elif self.layers[i].type == "pooling":
                spikes = self.layers[i].forward_pass(spikes, epoch)
            elif self.network_config["rule"] == "TSSLBP_DFA":
                spikes = self.layers[i].forward_pass(spikes, epoch, bypass=True, feedback=feedback)
            else:
                raise Exception('Unrecognized rule type. It is: {}'.format(self.network_config['rule']))
        return spikes


    def forward_sleep(self, layer_index):
        batch_size = self.network_config['batch_size']
        n_steps = self.network_config['n_steps']
        if self.layers[layer_index].type in ["dropout", "pooling"]:
            print("layer ", layer_index, "of type ", self.layers[layer_index].type, ": Skipping")
            return
        else:
            # print(self.layers[layer_index].in_shape)
            layer_inshape = self.layers[layer_index].in_shape
            # shape = (batch_size, layer_inshape[0], layer_inshape[1], layer_inshape[2], n_steps)
            shape = (batch_size, layer_inshape[0], layer_inshape[1], layer_inshape[2], n_steps)
            spike_input = torch.randn(shape).to(glv.device)
            if (self.network_config['sleep']['spike_value'] == "posonly"):
                spike_input[spike_input > 0] = 1
                spike_input[spike_input < 0] = 0
            else:
                spike_input[spike_input > 0] = 1
                spike_input[spike_input < 0] = -1


            if (self.network_config['sleep']['spike_input'] == "sparse"):
                sparse_generator = torch.rand(spike_input.shape)
                sparse_generator[sparse_generator < self.network_config['sleep']['sparse_level']] = 0
                sparse_generator[sparse_generator > 0] = 1
                spike_input = spike_input * sparse_generator.to(glv.device)
            spikes = f.psp(spike_input, self.network_config).to(glv.device)
            spike_trace = spikes
            for i in range(1, self.network_config['n_steps']):
                spike_trace[...,i] = spike_trace[...,i-1] + spikes[...,i]
            spikes = self.layers[layer_index].forward_pass(spikes, 0)





        for i in range(layer_index+1, len(self.layers)):
            if self.layers[i].type == "dropout":
                spikes = self.layers[i](spikes)
            elif self.network_config["rule"] in network_config_rules:
                spikes = self.layers[i].forward_pass(spikes, 0)
            else:
                raise Exception('Unrecognized rule type. It is: {}'.format(self.network_config['rule']))
        return spikes, spike_trace


    def get_parameters(self):
        return self.my_parameters


    def get_grads_dict(self):
        grads = {}
        for l in self.layers:
            filename = l.name + '_feedback_data.txt'
            # file = open('logs/' + filename, 'wa')
            # print(filename)
            # file.close()
            if (l.type == "conv" or l.type == "linear"):
                grads[l.name] = l.weight.grad
        return grads

    def get_grads_arr(self):
        grads = []
        for l in self.layers:
            if (l.type == "conv" or l.type == "linear"):
                grads.append(l.weight.grad.detach().cpu().numpy())
        return grads

    def get_dfa_params(self):
        dfas = []
        exceptions = 0
        for l in self.layers:
            try:
                dfas.extend(l.get_dfa_params())
            except:
                exceptions = exceptions + 1
                if exceptions > 1:
                    print("problem getting dfas")
                    # exit()
        return dfas
                # print("last layer")

    def get_TPA_params(self):
        matrices = []
        for l in self.layers:
            matrices.extend(l.get_TPA_params())
        return matrices

    def add_err(self, err):
        for l in self.layers:
            l.add_err(err)

    def TPA_train(self):
        for l in self.layers:
            l.TPA_train()

    def TPA_train_alt(self):
        for l in self.layers:
            l.TPA_train_alt()

    def TPA_train_alt_2(self):
        for l in self.layers:
            l.TPA_train_alt_2()

    def clear_grads(self):
        for l in self.layers:
            l.weight.grad.data.zero_()
            # l.weight.grad.data.zero_() = torch.empty_like(l.weight.grad)

    def weight_clipper(self):
        for l in self.layers:
            l.weight_clipper()

    def quantize_weights(self):
        for l in self.layers:
            l.quantize_weights()

    def weight_vary(self, amount=0.05):
        for l in self.layers:
            l.weight_vary(amount)

    def backward_KP(self, loss):
        temp_loss = loss
        backgrads = []
        for l in reversed(self.layers):
            temp_loss, backgrad = l.backward_KP(temp_loss)

            if (backgrad != None):
                # print(backgrad)
                tempgrad = l.kp_forward_hack(backgrad.detach())
                # loss = tempgrad.sum()
                # outloss = loss.backward()
                loss = self.local_loss.spike_kernel(torch.zeros_like(tempgrad), tempgrad, self.network_config)
                loss.backward()
                l.backward_matrix.grad = l.weight.grad


    def kp_decay(self):
        for l in self.layers:
            l.KP_decay()

    def get_stddev(self):
        for l in self.layers:
            print(torch.std(l.weight.grad))



    def kp_matrix_check(self, filename=''):
        for l in self.layers:
            l.kp_matrix_check(filename)