import argparse
import os
import pickle
import random
import sys
import time
from importlib import import_module

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from fjmp_dataloader_interaction import InteractionDataset as Dataset
from fjmp_dataloader_interaction import InteractionTestDataset as TestDataset
from fjmp_utils import *

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

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

def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data

def to_numpy(data):
    """Recursively transform torch.Tensor to numpy.ndarray.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_numpy(x) for x in data]
    if torch.is_tensor(data):
        data = data.numpy()
    return data

def to_int16(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_int16(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_int16(x) for x in data]
    if isinstance(data, np.ndarray) and data.dtype == np.int64:
        data = data.astype(np.int16)
    return data

def main():
    config = {}
    config['dataset_path'] = 'dataset_INTERACTION'
    config['tracks_train_reformatted'] = os.path.join(config['dataset_path'], 'train_reformatted')
    config['tracks_val_reformatted'] = os.path.join(config['dataset_path'], 'val_reformatted')
    config['tracks_test_reformatted'] = os.path.join(config['dataset_path'], 'test_reformatted')
    config['maps'] = 'dataset_INTERACTION/maps'
    config['num_scales'] = 6
    config["preprocess"] = False 
    config["val_workers"] = 0 
    config["workers"] = 0
    config['cross_dist'] = 10
    config['cross_angle'] = 1 * np.pi
    config["preprocess_train"] = os.path.join(config['dataset_path'], 'preprocess', 'train_interaction')
    config["preprocess_val"] = os.path.join(config['dataset_path'], 'preprocess', 'val_interaction')
    config["preprocess_test"] = os.path.join(config['dataset_path'], 'preprocess', 'test_interaction')
    config['batch_size'] = 1

    if not os.path.isdir(config["preprocess_train"]):
        os.makedirs(config["preprocess_train"])
    if not os.path.isdir(config["preprocess_val"]):
        os.makedirs(config["preprocess_val"])
    if not os.path.isdir(config["preprocess_test"]):
        os.makedirs(config["preprocess_test"])

    train(config)
    val(config)
    test(config)

def test(config):
    dataset = TestDataset(config)
    val_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    stores = [None for x in range(2644)]
    t = time.time()
    for i, data in enumerate(tqdm(val_loader)):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in ['idx', 
                        'city',
                        'track_ids',
                        'feats',
                        'ctrs',
                        'orig',
                        'theta',
                        'rot', 
                        'feat_locs',
                        'feat_vels', 
                        'feat_psirads',
                        'feat_shapes',
                        'feat_agenttopredicts',
                        'feat_interestingagents',
                        'feat_agenttypes',
                        'has_obss',
                        'graph']:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(stores, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader, config["preprocess_test"])

def val(config):
    dataset = Dataset(config, train=False)
    val_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    stores = [None for x in range(11794)]
    t = time.time()
    for i, data in enumerate(tqdm(val_loader)):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in ['idx', 
                        'city',
                        'feats',
                        'ctrs',
                        'orig',
                        'theta',
                        'rot', 
                        'feat_locs',
                        'feat_vels', 
                        'feat_psirads',
                        'feat_shapes',
                        'feat_agenttypes',
                        'gt_preds', 
                        'gt_vels', 
                        'gt_psirads',
                        'has_preds', 
                        'has_obss',
                        'ig_labels_sparse',
                        'ig_labels_dense',
                        'graph']:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(stores, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader, config["preprocess_val"])

def train(config):
    # Data loader for training set
    dataset = Dataset(config, train=True)
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    stores = [None for x in range(47584)]
    t = time.time()
    for i, data in enumerate(tqdm(train_loader)):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in ['idx', 
                        'city',
                        'feats',
                        'ctrs',
                        'orig',
                        'theta',
                        'rot', 
                        'feat_locs',
                        'feat_vels', 
                        'feat_psirads',
                        'feat_shapes',
                        'feat_agenttypes',
                        'gt_preds', 
                        'gt_vels', 
                        'gt_psirads',
                        'has_preds', 
                        'has_obss',
                        'ig_labels_sparse',
                        'ig_labels_dense',
                        'graph']:
                store[key] = to_numpy(data[key][j])
                # relevant graph data to int16 format
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()
    
    # apply ref_copy to graph
    dataset = PreprocessDataset(stores, config, train=True)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader, config["preprocess_train"])

class PreprocessDataset():
    def __init__(self, stores, config, train=True):
        self.stores = stores
        self.config = config
        self.train = train

    def __getitem__(self, idx):
        data = self.stores[idx]
        graph = dict()
        for key in ['lane_idcs', 'ctrs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'feats', 'centerlines', 'left_boundaries', 'right_boundaries']:
            graph[key] = ref_copy(data['graph'][key])
        graph['idx'] = idx
        # returns a subset of the graph information
        return graph

    def __len__(self):
        return len(self.stores)

def modify(config, data_loader, save):
    t = time.time()
    store = data_loader.dataset.stores
    for i, data in enumerate(data_loader):
        data = [dict(x) for x in data]

        out = []
        for j in range(len(data)):
            out.append(preprocess(to_long(gpu(data[j])), config['cross_dist'], config['cross_angle']))

        for j, graph in enumerate(out):
            idx = graph['idx']
            store[idx]['graph']['left'] = graph['left']
            store[idx]['graph']['right'] = graph['right']

        if (i + 1) % 100 == 0:
            print((i + 1) * config['batch_size'], time.time() - t)
            t = time.time()

        f = open(os.path.join(save, "{}.p".format(store[i]['idx'])), 'wb')
        pickle.dump(store[i], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

# From LaneGCN repository; This function mines the left/right neighbouring nodes
def preprocess(graph, cross_dist, cross_angle=None):
    # like pre and sec, but for left and right nodes
    left, right = dict(), dict()

    lane_idcs = graph['lane_idcs']
    # for each lane node lane_idcs returns the corresponding lane id
    num_nodes = len(lane_idcs)
    # indexing starts from 0, makes sense
    num_lanes = lane_idcs[-1].item() + 1

    # distances between all node centres
    dist = graph['ctrs'].unsqueeze(1) - graph['ctrs'].unsqueeze(0)
    dist = torch.sqrt((dist ** 2).sum(2))
    
    
    # allows us to index through all pairs of lane nodes
    # if num_nodes == 3: [0, 0, 0, 1, 1, 1, 2, 2, 2]
    hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    # if num_nodes == 3: [0, 1, 2, 0, 1, 2, 0, 1, 2]
    wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
    # if num_nodes == 3: [0, 1, 2]
    row_idcs = torch.arange(num_nodes).long().to(dist.device)

    # find possible left and right neighouring nodes
    if cross_angle is not None:
        # along lane
        f1 = graph['feats'][hi]
        # cross lane
        f2 = graph['ctrs'][wi] - graph['ctrs'][hi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = t2 - t1
        m = dt > 2 * np.pi
        dt[m] = dt[m] - 2 * np.pi
        m = dt < -2 * np.pi
        dt[m] = dt[m] + 2 * np.pi
        mask = torch.logical_and(dt > 0, dt < cross_angle)
        left_mask = mask.logical_not()
        mask = torch.logical_and(dt < 0, dt > -cross_angle)
        right_mask = mask.logical_not()

    # lanewise pre and suc connections
    pre = graph['pre_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
    suc = graph['suc_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

    # find left lane nodes
    pairs = graph['left_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        left_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        left_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            left_dist[hi[left_mask], wi[left_mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        left['u'] = ui.cpu().numpy().astype(np.int16)
        left['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        left['u'] = np.zeros(0, np.int16)
        left['v'] = np.zeros(0, np.int16)

    # find right lane nodes
    pairs = graph['right_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        right_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        right_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            right_dist[hi[right_mask], wi[right_mask]] = 1e6

        min_dist, min_idcs = right_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        right['u'] = ui.cpu().numpy().astype(np.int16)
        right['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        right['u'] = np.zeros(0, np.int16)
        right['v'] = np.zeros(0, np.int16)

    out = dict()
    out['left'] = left
    out['right'] = right
    out['idx'] = graph['idx']
    return out

main()

