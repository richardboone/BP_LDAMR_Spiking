import math

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init
import functions.TSSLBP as TSSLBP
import functions.TSSLBP_KP as TSSLBP_KP
import functions.DFA_fixed as DFAQ
import functions.TPA as TPA
import functions.loss_f as loss_f

from time import time 
import global_v as glv
import numpy as np
import qtorch
from qtorch.quant import fixed_point_quantize
import profiler

class LinearLayer(nn.Linear):
    def __init__(self, network_config, config, name, in_shape, device, dfafile=""):
        # extract information for kernel and inChannels
        in_features = config['n_inputs']
        out_features = config['n_outputs']
        self.layer_config = config
        self.network_config = network_config
        self.name = name
        self.type = config['type']
        self.in_shape = in_shape
        self.out_shape = [out_features, 1, 1]
        self.in_spikes = None
        self.out_spikes = None
        self.device = device
        # self.err_func = loss_f.SpikeLoss(network_config).to(device)

        if 'weight_scale' in config:
            weight_scale = config['weight_scale']
        else:
            weight_scale = 1

        if type(in_features) == int:
            n_inputs = in_features
        else:
            raise Exception('inFeatures should not be more than 1 dimesnion. It was: {}'.format(in_features.shape))
        if type(out_features) == int:
            n_outputs = out_features
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(out_features.shape))

        super(LinearLayer, self).__init__(n_inputs, n_outputs, bias=False)

        nn.init.kaiming_normal_(self.weight)
        self.weight = torch.nn.Parameter(weight_scale * self.weight, requires_grad=True)
        
        dfa_rules = ["TSSLBP_DFAQ", "TSSLBP_DFAS", "TEMPORAL_DFA", "DFA_SLEEP", "DFA_LT", "DFA_FT"]
        if ((self.network_config["rule"] in dfa_rules) and self.weight.shape[0] != network_config["n_class"]):
            self.dfa = True
        else:
            self.dfa = False

        if (self.network_config["rule"] == "TSSLBP_KP" or self.network_config["rule"] == "TPA"):
            self.decayrate = network_config["decayrate"]
            self.backward_matrix = torch.nn.Parameter((torch.randn(self.weight.shape)/10.0).squeeze(), requires_grad=True)
            self.backward_matrix.retain_grad()
            self.backward_matrix.requires_grad_(True)


        if (self.dfa):
            matshape = self.out_shape
            matshape.append(network_config["n_class"])
            if (network_config["dfa_genset"] == "randn"):
                self.dfa_matrix = torch.randn(matshape, device=device, requires_grad=True).squeeze()
            elif (network_config["dfa_genset"] == "rand"):
                self.dfa_matrix = (torch.rand(matshape, device=device, requires_grad=True).squeeze() - 0.5)
            elif (network_config["dfa_genset"] == "rand2"):
                self.dfa_matrix = (torch.rand(matshape, device=device, requires_grad=True).squeeze() - 0.5) * 2
            else:
                print("don't recognize dfa setting, defaulting to randn")
                self.dfa_matrix = torch.randn(matshape, device=device, requires_grad=True).squeeze()

            if (self.network_config["rule"] == "DFA_FT"):
                if (self.network_config["dfa_learn_preset"] == "none"):
                    print("not ready yet")
                    #nothing
                elif (self.network_config["dfa_learn_preset"] == "triangular"):
                    print("not there yet")
                    temp_mat = glv.partial_a.clone()
                    print(temp_mat.shape)
                    temp_mat[temp_mat > 0] = 1
                    self.dfa_matrix = torch.einsum('abcd, cd -> abcd', self.dfa_matrix, temp_mat.squeeze())
                elif (self.network_config["dfa_learn_preset"] == "partial_a"):
                    matshape.pop()
                    matshape.pop()

                    self.dfa_matrix = torch.randn(matshape, device=device, requires_grad=True).squeeze()
                    temp_mat = glv.partial_a.clone()
                    self.dfa_matrix = torch.einsum('ab, cd -> abcd', self.dfa_matrix, temp_mat.squeeze())





        print("linear")
        print(self.name)
        print(self.in_shape)
        if (self.dfa):
            print(self.dfa_matrix.shape)
        print(self.out_shape)

        print(list(self.weight.shape))
        print("-----------------------------------------")

    def forward(self, x):
        """
        """
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = x.transpose(1, 2)
        y = f.linear(x, self.weight, self.bias)
        y = y.transpose(1, 2)
        y = y.view(y.shape[0], y.shape[1], 1, 1, y.shape[2])
        return y

    def forward_pass(self, x, epoch, bypass=False, feedback=None, bp_pass=False):
        self.inputs = x.detach().clone().to(glv.device)

        if (self.network_config["rule"] != "TSSLBP_KP" and self.network_config["rule"] != "TPA"):
            # print(type(x))
            y = self.forward(x)
        elif self.network_config["rule"] == "TSSLBP_KP":
            y = TSSLBP_KP.KP_linear_autograd.apply(x, self.network_config, self.layer_config, self.weight, self.bias, self.backward_matrix)
        elif self.network_config["rule"] == "TPA":
            if (self.network_config["tpa_bw"] == True):
                y = TPA.TPA_linear_layer.apply(x, self.network_config, self.layer_config, self.weight, self.bias, self.weight)
            else:
                y = TPA.TPA_linear_layer.apply(x, self.network_config, self.layer_config, self.weight, self.bias, self.backward_matrix)
        shape = x.shape


        if(self.network_config["rule"]=="TSSLBP"):
            if shape[4] > shape[0] * 10:
                y = TSSLBP.PSP_spike_long_time.apply(y, self.network_config, self.layer_config)
            else:
                y = TSSLBP.PSP_spike_large_batch.apply(y, self.network_config, self.layer_config)
        elif self.network_config["rule"] == "TSSLBP_DFAQ":
            if (self.dfa):
                if shape[4] > shape[0] * 10:
                    y = DFAQ.PSP_spike_long_time.apply(y, self.network_config, self.layer_config, self.dfa_matrix)
                else:
                    y = DFAQ.PSP_spike_large_batch.apply(y, self.network_config, self.layer_config, self.dfa_matrix)
            else:
                if shape[4] > shape[0] * 10:
                    y = TSSLBP.PSP_spike_long_time.apply(y, self.network_config, self.layer_config)
                else:
                    y = TSSLBP.PSP_spike_large_batch.apply(y, self.network_config, self.layer_config)

        elif (self.network_config["rule"] == "TSSLBP_KP"):
            if shape[4] > shape[0] * 10:
                y = TSSLBP.PSP_spike_long_time.apply(y, self.network_config, self.layer_config)
            else:
                y = TSSLBP.PSP_spike_large_batch.apply(y, self.network_config, self.layer_config)
        elif (self.network_config["rule"] == "TPA"):
            y, self.est_delta_u, self.temp_outputs= TPA.PSP_spike_large_batch.apply(y, self.network_config, self.layer_config)


        self.output = y
        return y

    def get_parameters(self):
        if (self.network_config["rule"] == "TSSLBP_KP"):
            return [self.weight, self.backward_matrix]
        else:
            return [self.weight]

    def get_dfa_params(self):
        if (self.dfa):
            try:
                return [self.dfa_matrix, self.temporal_matrix]
            except:
                return [self.dfa_matrix]
        else:
            return []

    def get_TPA_params(self):
        return [self.backward_matrix]

    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w
        if (self.dfa):
            dfm = self.dfa_matrix.data
            dfm = dfm.clamp(-4,4)
            self.dfa_matrix.data = dfm
        if (self.network_config["rule"] == "TSSLBP_KP" or self.network_config["rule"] == "TPA"):
            tw = self.backward_matrix.data
            tw = tw.clamp(-4,4)
            self.backward_matrix.data = tw


    def TPA_train_alt_2(self):

        # inputs = self.inputs.detach().clone().to(glv.device)
        threshold = self.layer_config["threshold"]
        inputs = self.inputs
        inputs.requires_grad = True

        stddevs = torch.std(inputs, (1,2,3,4), unbiased=True).to(glv.device)
        perturb = torch.einsum('abcde, a -> abcde', torch.rand(inputs.shape, device=glv.device),  stddevs) * self.network_config['tpa_perturbation']

        new_input = inputs + perturb
        old_outputs = self.output.clone().detach().to(glv.device)

        new_output = self.forward_pass(new_input, 0).detach().clone().to(glv.device)
        new_output.requires_grad = True
        diff = new_output - old_outputs

        #######################fake backward here#####################

        shape = diff.shape

        
        grad = torch.einsum('...ij, ...j -> ...i', glv.partial_a.repeat(shape[0], shape[1], shape[2], shape[3], 1, 1), diff) *  (torch.clamp(1 / self.est_delta_u, -10, 10) * self.temp_outputs)
        
        #######################fake backward end#######################


        est_out1 = TPA.TPA_back_linear.apply(grad, self.backward_matrix.transpose(0,1))
        
        part_1 = perturb.view((perturb.shape[0], perturb.shape[1]*perturb.shape[2]*perturb.shape[3], perturb.shape[4])) - est_out1.view((est_out1.shape[0], est_out1.shape[1]*est_out1.shape[2]*est_out1.shape[3], est_out1.shape[4]))
        real_grad = torch.einsum('abcde, afe -> fb', grad, part_1)
        self.backward_matrix.grad = real_grad.transpose(0,1).detach()
        return
        


    def add_err(self, err):
        self.err = err

    def KP_decay(self):
        weight_decay = self.weight.data * self.decayrate
        matrix_decay = self.backward_matrix.data * self.decayrate
        self.weight.data = self.weight.data - weight_decay
        self.backward_matrix.data = self.backward_matrix.data - matrix_decay
        return







