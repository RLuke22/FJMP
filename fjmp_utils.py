import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.distributions import kl_divergence
import numpy as np 
import random
import horovod.torch as hvd 
import sys, os, math
import matplotlib.pyplot as plt
from scipy import sparse

def accumulate_gradients(grads, named_parameters):
    if grads == {}:
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                grads[n] = p.grad.abs().mean()
    else:
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                grads[n] += p.grad.abs().mean()
    
    return grads

def plot_grad_flow(grads, epoch, log_path):
    path = log_path / 'gradients_{}.png'.format(epoch)
    plt.rc('xtick', labelsize=4)
    plt.figure(figsize=(20, 20), dpi=200)

    to_plot = list(grads.values())
    to_plot = [x.detach().cpu() for x in to_plot]
    
    plt.plot(to_plot, alpha=0.3, color="b")
    plt.hlines(0, 0, len(grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(grads), 1), list(grads.keys()), rotation="vertical")
    plt.xlim(xmin=0, xmax=len(grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(path)
    print("Plotted gradient flow")

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

def sync3(data, comm):
    data_list = comm.allgather(data)
    
    final_grads = {}
    for i in range(len(data_list)):
        if i == 0:
            for key in data_list[i].keys():
                final_grads[key] = data_list[i][key]
        else:
            for key in data_list[i].keys():
                final_grads[key] += data_list[i][key]
    
    for key in final_grads.keys():
        final_grads[key] /= len(data_list)
    
    return final_grads

def sync(data, config, comm):
    data_list = comm.allgather(data)
    
    FDE = 0
    ADE = 0
    SCR = 0
    SMR = 0
    SMR_AV2 = 0
    pFDE = 0
    pADE = 0
    n_scenarios = 0
    for i in range(len(data_list)):
        FDE += data_list[i]['FDE'] * data_list[i]['n_scenarios']
        ADE += data_list[i]['ADE'] * data_list[i]['n_scenarios']
        SCR += data_list[i]['SCR'] * data_list[i]['n_scenarios']
        SMR += data_list[i]['SMR'] * data_list[i]['n_scenarios']
        SMR_AV2 += data_list[i]['SMR_AV2'] * data_list[i]['n_scenarios']
        pFDE += data_list[i]['pFDE'] * data_list[i]['n_scenarios']
        pADE += data_list[i]['pADE'] * data_list[i]['n_scenarios']
        n_scenarios += data_list[i]['n_scenarios']
    
    FDE /= n_scenarios
    ADE /= n_scenarios
    SCR /= n_scenarios
    SMR /= n_scenarios
    SMR_AV2 /= n_scenarios
    pFDE /= n_scenarios
    pADE /= n_scenarios

    if config['learned_relation_header']:
        n_gpus = 0
        edge_acc = 0
        edge_acc_0 = 0
        edge_acc_1 = 0
        edge_acc_2 = 0
        proportion_no_edge = 0
        for i in range(len(data_list)):
            n_gpus += 1
            edge_acc += data_list[i]['Edge Accuracy']
            edge_acc_0 += data_list[i]['Edge Accuracy 0']
            edge_acc_1 += data_list[i]['Edge Accuracy 1']
            edge_acc_2 += data_list[i]['Edge Accuracy 2']
            proportion_no_edge += data_list[i]['Proportion No Edge']
        
        edge_acc /= n_gpus
        edge_acc_0 /= n_gpus
        edge_acc_1 /= n_gpus
        edge_acc_2 /= n_gpus
        proportion_no_edge /= n_gpus
    
    return_dict = {
        'FDE': FDE,
        'ADE': ADE,
        'pFDE': pFDE,
        'pADE': pADE,
        'SCR': SCR,
        'SMR': SMR,
        'SMR_AV2': SMR_AV2
    }

    if config["learned_relation_header"]:
        return_dict['E-Acc'] = edge_acc
        return_dict['E-Acc 0'] = edge_acc_0 
        return_dict['E-Acc 1'] = edge_acc_1 
        return_dict['E-Acc 2'] = edge_acc_2 
        return_dict['PropNoEdge'] = proportion_no_edge
    
    return return_dict

class Logger(object):
    def __init__(self, log):
        self.terminal = sys.stdout
        self.log = open(log, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

### FROM LANE_GCN
def graph_gather(graphs, config):
    batch_size = len(graphs)
    node_idcs = []
    count = 0
    counts = []

    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = dict()
    graph["idcs"] = node_idcs
    graph["ctrs"] = [x["ctrs"] for x in graphs]

    for key in ["feats"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)

    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(min(len(graphs[0]["pre"]), config["num_scales"])):
            graph[k1].append(dict())
            for k2 in ["u", "v"]:
                graph[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                )

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [
                x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                for x in temp
            ]
            graph[k1][k2] = torch.cat(temp)
    
    return graph

### FROM LANE_GCN
def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)
    return nbrs

### FROM LANE_GCN
def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data

### FROM LANE_GCN
def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data

### FROM LANE_GCN
def cat(batch):
    if torch.is_tensor(batch[0]):
        batch = [x.unsqueeze(0) for x in batch]
        return_batch = torch.cat(batch, 0)
    elif isinstance(batch[0], list) or isinstance(batch[0], tuple):
        batch = zip(*batch)
        return_batch = [cat(x) for x in batch]
    elif isinstance(batch[0], dict):
        return_batch = dict()
        for key in batch[0].keys():
            return_batch[key] = cat([x[key] for x in batch])
    else:
        return_batch = batch
    return return_batch

### FROM LANE_GCN
def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch

### FROM LANE_GCN
def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data

### FROM LANE_GCN
def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data

def print_(*args):
    if hvd.rank() == 0:
        print(*args)

def worker_init_fn(pid):
    np_seed = hvd.rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)

### FROM NRI
def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)

### FROM CONTRASTIVE FUTURE TRAJECTORY PREDICTION
def estimate_constant_velocity(history, prediction_horizon, has_obs):
    history = history[has_obs == 1]
    length_history = history.shape[0]
    z_x = history[:, 0] # these are the observations x
    z_y = history[:, 1] # these are the observations y
    
    if length_history == 1:
        v_x = 0
        v_y = 0
    else:
        v_x = 0
        v_y = 0
        for index in range(length_history - 1):
            v_x += z_x[index + 1] - z_x[index]
            v_y += z_y[index + 1] - z_y[index]
        v_x = v_x / (length_history - 1) # v_x is the average velocity x
        v_y = v_y / (length_history - 1) # v_y is the average velocity y
    
    x_pred = z_x[-1] + v_x * prediction_horizon 
    y_pred = z_y[-1] + v_y * prediction_horizon 

    return x_pred, y_pred

def evaluate_fde(x_pred, y_pred, x, y):
    return math.sqrt((x_pred - x) ** 2 + (y_pred - y) ** 2)

class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )

def sign_func(x):
    if x > 0:
        return 1.
    elif x < 0:
        return -1.
    else:
        return 0.