import torch
import torch.nn as nn
import torch.nn.functional as f
from time import time 
import global_v as glv


def psp(inputs, network_config):
    shape = inputs.shape
    n_steps = network_config['n_steps']
    tau_s = network_config['tau_s']

    syn = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=glv.dtype, device=glv.device)
    syns = torch.zeros((shape[0], shape[1], shape[2], shape[3], n_steps), dtype=glv.dtype, device=glv.device)

    for t in range(n_steps):
        syn = syn - syn / tau_s + inputs[..., t]
        syns[..., t] = syn / tau_s

    return syns



class TPA_back_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        temp_inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3], inputs.shape[4])
        temp_inputs = temp_inputs.transpose(1,2)
        outputs = f.linear(temp_inputs, weights)
        outputs = outputs.transpose(1,2)
        outputs = outputs.view(outputs.shape[0], outputs.shape[1], 1, 1, outputs.shape[2])


        ctx.save_for_backward(inputs, weights)
        return outputs

    def backward(ctx, grad_delta):
        (inputs, forward_weights) = ctx.saved_tensors
        #weight gradients here


        temp_inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3], inputs.shape[4])

        temp_grads = grad_delta.view(grad_delta.shape[0], grad_delta.shape[1] * grad_delta.shape[2] * grad_delta.shape[3], grad_delta.shape[4])

        weight_grad = torch.einsum('abc, afc -> bf', temp_grads, temp_inputs)
        weight_grad = torch.clamp(weight_grad, -100, 100)
        return None, weight_grad


class TPA_linear_layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, network_config, layer_config, forward_weights, forward_bias, backward_weights):
        temp_inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3], inputs.shape[4])
        temp_inputs = temp_inputs.transpose(1,2)
        outputs = f.linear(temp_inputs, forward_weights)
        outputs = outputs.transpose(1,2)
        outputs = outputs.view(outputs.shape[0], outputs.shape[1], 1, 1, outputs.shape[2])


        ctx.save_for_backward(inputs, forward_weights, backward_weights, forward_bias)
        return outputs

    @staticmethod
    def backward(ctx, grad_delta):
        (inputs, forward_weights, backward_weights, forward_bias) = ctx.saved_tensors
        #weight gradients ehre

        temp_inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3], inputs.shape[4])

        temp_grads = grad_delta.view(grad_delta.shape[0], grad_delta.shape[1] * grad_delta.shape[2] * grad_delta.shape[3], grad_delta.shape[4])

        weight_grad = torch.einsum('abc, afc -> bf', temp_grads, temp_inputs)



        outputs = torch.einsum('abc, db -> adc', temp_grads, backward_weights.transpose(0,1))

        outputs = outputs.view(inputs.shape)
        outputs = torch.clamp(outputs, -100, 100)
        weight_grad = torch.clamp(weight_grad, -100, 100)
        return outputs, None, None,weight_grad, None,None


class PSP_spike_large_batch(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, inputs, network_config, layer_config):
        shape = inputs.shape
        n_steps = network_config['n_steps']
        theta_m = 1/network_config['tau_m']
        theta_s = 1/network_config['tau_s']
        threshold = layer_config['threshold']

        mem = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=glv.dtype, device=glv.device)
        syn = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=glv.dtype, device=glv.device)
        ref = torch.zeros((shape[0], shape[1], shape[2], shape[3], shape[4]), dtype=glv.dtype, device=glv.device)
        delta_refs = torch.zeros((shape[0], shape[1], shape[2], shape[3], shape[4], shape[4]), dtype=glv.dtype, device=glv.device)
        est_mem_updates = torch.zeros((shape[0], shape[1], shape[2], shape[3], shape[4]), dtype=glv.dtype, device=glv.device)
        spike_dist = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=torch.long, device=glv.device)
        # prev_spikes  = prev_spikes - 1
        gen = torch.arange(shape[4], device=glv.device).repeat((shape[0], shape[1], shape[2], shape[3], 1))
        est_delta_u = torch.zeros((shape[0], shape[1], shape[2], shape[3], shape[4]), dtype=glv.dtype, device=glv.device)
        mems = []
        mem_updates = []
        outputs = []
        syns_posts = []
        for t in range(n_steps):
            mem_update = (-theta_m) * mem + inputs[..., t]
            delta_ref = (-theta_m) * ref
            mem += mem_update
            ref += delta_ref

            out = mem > threshold
            out = out.type(glv.dtype)


            #########################################mem update estimation section

            est_mem_updates[...,t] = 1

            # denoms = glv.partial_currents[spike_dist]
            adjust  = torch.einsum('abcde, abcd -> abcde', est_mem_updates, torch.mul(out,glv.partial_currents[spike_dist]))
            # print(adjust.shape)
            # est_mem_updates *= (1-out)*(-1)
            est_mem_updates  = torch.einsum('abcd, abcde -> abcde', (out-1)*(-1), est_mem_updates)

            est_mem_updates = est_mem_updates - (theta_m * est_mem_updates)
            
            spike_dist = torch.mul(spike_dist + 1, (out -1)*(-1)).long()

            est_delta_u += adjust

           

            ################################



            mems.append(mem)
            if t > 0:
                out_tmp = out.unsqueeze(-1).repeat(1, 1, 1, 1, t)
                ref[..., 0:t] *= (1-out_tmp)
                delta_ref[..., 0:t] *= out_tmp
            ref[..., t] = (-1) * mem * out
            delta_refs[..., 0:t, t] = delta_ref[..., 0:t]

            mem = mem * (1-out)
            outputs.append(out)
            mem_updates.append(mem_update)

            syn = syn + (out - syn) * theta_s
            syns_posts.append(syn)
        mems = torch.stack(mems, dim = 4)
        mem_updates = torch.stack(mem_updates, dim = 4)
        syns_posts = torch.stack(syns_posts, dim = 4)
        outputs = torch.stack(outputs, dim = 4)

 


        ##filler options##
        
        if (network_config["tpa_filler"] == "low"):#padd ending as if lowest filler level
            est_mem_updates *= glv.partial_currents[-1]/2
            est_delta_u += est_mem_updates
        elif (network_config["tpa_filler"] == "avg"):
            # print("nah")
            num_spikes = torch.sum(outputs, dim=4)
            timediff = shape[-1] - spike_dist
            vals = torch.nan_to_num(num_spikes/timediff)
            est_mem_updates = torch.einsum('abcde, abcd -> abcde', est_mem_updates, vals)
            est_delta_u += est_mem_updates
        elif (network_config["tpa_filler"] == "avg_low"):
            # print("nah")
            num_spikes = torch.sum(outputs, dim=4)
            timediff = shape[-1] - spike_dist
            vals = torch.nan_to_num(num_spikes/timediff, nan=-1 * glv.partial_currents[-1]/2)
            est_mem_updates = torch.einsum('abcde, abcd -> abcde', est_mem_updates, vals)
            est_delta_u += est_mem_updates

        elif (network_config["tpa_filler"] == "final_spike"):
            # print("??")
            est_mem_updates  = torch.einsum('abcde, abcd -> abcde', est_mem_updates, glv.partial_currents[torch.clamp(spike_dist, max=shape[-1]-1)])
            est_delta_u += est_mem_updates
        elif (network_config["tpa_filler"] == "empty_diff"):
            est_mem_updates *= -1 * glv.partial_currents[-1]/2
            est_delta_u += est_mem_updates
        elif (network_config["tpa_filler"] == "diff_all"):
            negate = torch.ones_like(spike_dist)
            negate[spike_dist == shape[-1]] = -1
            est_mem_updates  = torch.einsum('abcde, abcd -> abcde', est_mem_updates, glv.partial_currents[torch.clamp(spike_dist, max=shape[-1]-1)])
            est_mem_updates = torch.einsum('abcde, abcd -> abcde', est_mem_updates, negate)
            est_delta_u += est_mem_updates

         ##end filler##
        est_delta_u *= threshold

        if(network_config["tpa_bg"]):
            est_delta_u = mem_updates
        ctx.save_for_backward(mem_updates, outputs, mems, est_delta_u, torch.tensor([threshold]))
        return syns_posts, est_delta_u, outputs

    @staticmethod
    def backward(ctx, grad_delta, du_grads, dontcare):
        (delta_u, outputs, u, est_delta_u, others) = ctx.saved_tensors
        start_time = time()
        shape = outputs.shape
        n_steps = glv.n_steps
        threshold = others[0].item()

        mini_batch = shape[0]
        partial_a_inter = glv.partial_a.repeat(mini_batch, shape[1], shape[2], shape[3], 1, 1)
        grad_a = torch.empty_like(delta_u)


        for i in range(int(shape[0]/mini_batch)):
            partial_a_all = partial_a_inter

            grad_a[i*mini_batch:(i+1)*mini_batch, ...] = torch.einsum('...ij, ...j -> ...i', partial_a_all, grad_delta[i*mini_batch:(i+1)*mini_batch, ...])

        if torch.sum(outputs)/(shape[0] * shape[1] * shape[2] * shape[3] * shape[4]) > 0.1:
            partial_u = torch.clamp(1 / est_delta_u, -10, 10) * outputs
            grad = grad_a * partial_u
        else:
            # warm up
            a = 0.2
            f = torch.clamp((-1 * u + threshold) / a, -8, 8)
            f = torch.exp(f)
            f = f / ((1 + f) * (1 + f) * a)

            grad = grad_a * f
        return grad, None, None, None, None, None, None, None, None


