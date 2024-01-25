import torch


dtype = None
device = None
n_steps = None
partial_a = None
tau_s = None
dfa_feedback = None
output_spikes = None
extracted_matrix = None
a = None
samples = None
errors = None
labels = None
max_partial_a = None
limit_time = None
output_logs = None
dfa_error = None
profile = False
partial_currents = None

def sum_tau(dist, tau_m):
    sum = 0
    for i in range(dist):
        sum += tau_m**i
    return sum

def current_eval(dist, tau_m):
    top = (2*dist) - 1
    if (dist == 1):
        bottom = 3
    else:
        bottom = 2*(dist-1)
    right = top/bottom
    print(dist, tau_m, sum_tau(dist, tau_m))
    current = right*sum_tau(dist, tau_m)
    return current


def init(dty, dev, n_t, ts, lt=0, network_config=None):   # a(t_k) = (1/tau)exp(-(t_k-t_m)/tau)
    global dtype, device, n_steps, partial_a, tau_s, dfa_feedback, a, max_partial_a, limit_time, output_spikes, dfa_error, profile, partial_currents
    dfa_error = []
    dtype = dty
    device = dev
    n_steps = n_t
    tau_s = ts
    partial_a = torch.zeros((1, 1, 1, 1, n_steps, n_steps), dtype=dtype).to(device)
    max_partial_a = torch.zeros((1, 1, 1, 1, n_steps*2, n_steps*2), dtype=dtype).to(device)
    partial_currents = torch.zeros(n_steps, dtype=dtype).to(device)
    a = 0
    for t in range(n_steps):
        if(network_config != None):
            if (network_config["tpa_vcurve"] == "large_flat"):
                partial_currents[t] = (1+(1/(t+1)))/(t+1)
            elif (network_config["tpa_vcurve"] == "flat"):
                partial_currents[t] = (1)/(t+1)
            elif (network_config["tpa_vcurve"] == "abbrev_theory"):
                partial_currents[t] = current_eval(t+1, network_config["tau_m"])
            else:
                pass
        else:
            pass
        if (t > 0):
            partial_a[..., t] = partial_a[..., t - 1] - partial_a[..., t - 1] / tau_s 
        partial_a[..., t, t] = 1/tau_s


    print(partial_currents)
    # exit()
    for t in range(n_steps*2):
        if t > 0:
            max_partial_a[..., t] = max_partial_a[..., t - 1] - max_partial_a[..., t - 1] / tau_s 
        max_partial_a[..., t, t] = 1/tau_s



    # partial_a = torch.einsum('...ij, ...ij -> ...ij', partial_a, partial_a)
    print(partial_a[...,0:5,0:5])
    # print(max_partial_a)
    # exit()
    if (lt != 0):
        limit_time = lt
        minval = partial_a[0,0,0,0,0,lt-1]
        vals = partial_a[0,0,0,0,0,0:lt]
        partial_a[partial_a < minval] = 0


def add_samples(somestuff):
    global samples
    print("samples")
    if (samples == None):
        samples = somestuff
    else:
        samples.append(somestuff)


def add_errors(new_errors):
    global errors
    # print("samples")
    if (errors == None):
        errors = new_errors
    else:
        errors.append(new_errors)

def add_labels(new_labels):
    global labels
    if (labels == None):
        labels = new_labels
    else:
        labels.append(new_labels)



def export_files(filename_list):
    print("not yet")

def check_values():
    global errors, samples, labels
    print(samples.shape)
    print(errors.shape)
    print(labels.shape)