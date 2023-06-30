import numpy as np
import torch
from torch.utils.data import Dataset
import os
import copy
import csv
import pickle
import re
from pandas import read_csv
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from fjmp_utils import *
from fjmp_metrics import *
from pathlib import Path

np.set_printoptions(suppress=True)

import av2
from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet, _convert_tracks_to_tabular_format
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.interpolate import compute_midpoint_line
from scipy import sparse

class Argoverse2Dataset(Dataset):
    def __init__(self, config, train=True, train_all=False):
        self.config = config 
        self.train = train 

        if self.train:
            self.preprocess_path = self.config["preprocess_train"]
            self.mapping_filename = 'mapping_train_argoverse2.pkl'
            self.files = self.config["files_train"]
            if train_all:
                self.n_samples = 199908 + 24988
            else:
                self.n_samples = 199908
        else:
            self.preprocess_path = self.config["preprocess_val"]
            self.mapping_filename = 'mapping_val_argoverse2.pkl'
            self.files = self.config["files_val"]
            self.n_samples = 24988

        # load mapping dictionary
        with open(os.path.join(self.config['dataset_path'], self.mapping_filename), "rb") as f:
            self.mapping = pickle.load(f)

        self.avg_agent_length = {
            0: 4.0,
            1: 0.7,
            2: 2.0,
            3: 2.0,
            4: 12.0
        }

        self.avg_agent_width = {
            0: 2.0,
            1: 0.7,
            2: 0.7,
            3: 0.7,
            4: 2.5
        }

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.config['preprocess']:
            data = np.load(os.path.join(self.preprocess_path, "{}.p".format(idx)), allow_pickle=True)
            new_data = dict()
            for key in ['idx', # unique integer assigned to each scene (assigned when building mapping dictionary)
                        'orig', # origin of scene (taken to be present location of random vehicle in scene during training and of ego vehicle during validation)
                        'feats', # position features (offsets) in SE(2)-transformed coordinate system (past + future)
                        'feat_locs', # location features (not offsets but absolute positions in SE(2)-transformed coordinate system) (past + future)
                        'feat_vels', # velocity features, in SE(2)-transformed coordinate system (past + future)
                        'feat_agenttypes', # agent_type (either car, pedestrian, bicycle, motorcyclist, bus, past + future)
                        'feat_agentcategories', # either unscored_track, scored_track, or focal_track
                        'feat_psirads', # yaw angle features, in SE(2)-transformed coordinate system (past + future)
                        'gt_preds', # ground-truth positions (future)
                        'gt_vels', # ground-truth velocities (future)
                        'gt_psirads', # ground-truth yaw angles (future)
                        'has_preds', # future timestep exists mask (future)
                        'has_obss', # past timestep exists mask (past)
                        'theta', # angle for rotating scene
                        'rot', # rotation matrix for rotating scene
                        'ctrs', # agent centers at the present timestep in SE(2)-transformed coordinate system
                        'ig_labels_sparse', # interaction graph labels for current scene (eps_I = 2.5s)
                        'ig_labels_dense', # interaction graph labels for current scene (eps_I = 6.0s)
                        'ig_labels_m2i', # interaction graph labels for current scene (M2I heuristic)
                        'graph']: # lane graph

                if key in data:
                    new_data[key] = ref_copy(data[key])
            
            data = new_data 

            return data
        
        # otherwise we process the data
        data = self.read_argoverse2_data(idx)
        data = self.get_obj_feats(data, idx)
        data['idx'] = idx 
        data['graph'] = self.get_lane_graph(data, idx)

        # not needed for downstream processing in preprocess
        del data['trajs']
        del data['steps']
        del data['vels']
        del data['psirads']
        del data['agenttypes']
        del data['agentcategories']
        if self.train:
            del data['track_ids']
        del data['is_valid_agent']

        return data

    def read_argoverse2_data(self, idx):
        scene_directory = self.mapping[idx]
        parquet_file = os.path.join(self.files, scene_directory, "scenario_{}.parquet".format(scene_directory))
        scenario = load_argoverse_scenario_parquet(parquet_file)
        
        """observed, track_id, object_type, object_category, timestep, position_x, position_y, heading, velocity_x, velocity_y"""
        df = _convert_tracks_to_tabular_format(scenario.tracks)
        
        agt_ts = np.sort(np.unique(df['timestep'].values))
        timestamp_mapping = dict()
        for i, ts in enumerate(agt_ts):
            timestamp_mapping[ts] = i 

        trajs = np.concatenate((
            df.position_x.to_numpy().reshape(-1, 1),
            df.position_y.to_numpy().reshape(-1, 1)
        ), 1)

        vels = np.concatenate((
            df.velocity_x.to_numpy().reshape(-1, 1),
            df.velocity_y.to_numpy().reshape(-1, 1)
        ), 1)

        psirads = df.heading.to_numpy().reshape(-1, 1)

        track_ids = df.track_id.to_numpy().reshape(-1, 1)

        agentcategories = df.object_category.to_numpy().reshape(-1, 1)

        ### NOTE: We will only predict trajectories from classes 0-4
        object_type_dict = {
            'vehicle': 0,
            'pedestrian': 1,
            'motorcyclist': 2,
            'cyclist': 3,
            'bus': 4,
            'static': 5,
            'background': 6,
            'construction': 7,
            'riderless_bicycle': 8,
            'unknown': 9
        }

        agenttypes = []
        for x in df.object_type:
            agenttypes.append(object_type_dict[x])
        agenttypes = np.array(agenttypes).reshape(-1, 1)

        ### NOTE: no shape information in Argoverse 2.

        steps = [timestamp_mapping[x] for x in df['timestep'].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['track_id']).groups 
        keys = list(objs.keys())
        ctx_trajs, ctx_steps, ctx_vels, ctx_psirads, ctx_agenttypes, ctx_agentcategories, ctx_track_ids = [], [], [], [], [], [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])
            ctx_vels.append(vels[idcs])
            ctx_psirads.append(psirads[idcs])
            ctx_agenttypes.append(agenttypes[idcs])  
            ctx_agentcategories.append(agentcategories[idcs])
            ctx_track_ids.append(track_ids[idcs])

        data = dict()
        data['trajs'] = ctx_trajs
        data['steps'] = ctx_steps 
        data['vels'] = ctx_vels
        data['psirads'] = ctx_psirads
        data['agenttypes'] = ctx_agenttypes
        data['agentcategories'] = ctx_agentcategories
        data['track_ids'] = ctx_track_ids

        return data

    def get_obj_feats(self, data, idx):
        if self.train:
            orig_idx =  idx % len(data['trajs'])
            while True:
                # Are the observed timesteps available for this agent?
                found = True
                for i in range(50):
                    if i not in data['steps'][orig_idx]:
                        found = False
                        break
                if found:
                    break
                else:
                    orig_idx = (orig_idx + 1) % len(data['trajs'])
        else:
            found_AV = False
            for i in range(len(data['track_ids'])):
                if 'AV' in data['track_ids'][i]:
                    found_AV = True
                    break
            
            assert found_AV
            assert len(data['track_ids'][i]) == 110

            orig_idx = i
            del data['track_ids']
        
        orig = data['trajs'][orig_idx][49].copy().astype(np.float32)
        pre = data['trajs'][orig_idx][48] - orig 
        # Since theta is pi - arctan(.), then the range of theta is
        # max: pi - (-pi) = 2pi
        # min: pi - (pi) = 0
        theta = np.pi - np.arctan2(pre[1], pre[0])

        # rotation matrix for rotating scene
        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        feats, feat_locs, feat_vels, gt_preds, gt_vels, has_preds, has_obss = [], [], [], [], [], [], []
        feat_psirads, gt_psirads, ctrs, feat_agenttypes, feat_agentcategories = [], [], [], [], []
        is_valid_agent = []

        for traj, step, vel, psirad, agenttype, agentcategory in zip(data['trajs'], data['steps'], data['vels'], data['psirads'], data['agenttypes'], data['agentcategories']):
            if 49 not in step:
                is_valid_agent.append(0)
                continue

            # if not a dynamic vehicle
            if agenttype[0, 0] >= 5:
                is_valid_agent.append(0)
                continue

            # ignore track fragments
            if agentcategory[0, 0] == 0:
                is_valid_agent.append(0)
                continue

            is_valid_agent.append(1)

            # ground-truth future positions
            gt_pred = np.zeros((60, 2), np.float32)
            # ground truth future velocities
            gt_vel = np.zeros((60, 2), np.float32)
            # ground truth yaw angles
            gt_psirad = np.zeros((60, 1), np.float32)

            # has ground-truth future mask
            has_pred = np.zeros(60, bool)
            has_obs = np.zeros(110, bool)

            future_mask = np.logical_and(step >= 50, step < 110)
            post_step = step[future_mask] - 50
            post_traj = traj[future_mask]
            post_vel = vel[future_mask]
            post_agenttype = agenttype[future_mask]
            post_psirad = psirad[future_mask]
            gt_pred[post_step] = post_traj
            gt_vel[post_step] = post_vel
            gt_psirad[post_step] = post_psirad
            has_pred[post_step] = 1

            # observation + future horizon
            idcs = step.argsort()
            step = step[idcs]
            traj = traj[idcs]
            vel = vel[idcs]
            agenttype = agenttype[idcs]
            psirad = psirad[idcs]
            agentcategory = agentcategory[idcs]
            has_obs[step] = 1

            # only observation horizon
            obs_step = step[step < 50]
            obs_idcs = obs_step.argsort()
            obs_step = obs_step[obs_idcs]

            # take contiguous past to be the past
            for i in range(len(obs_step)):
                if obs_step[i] == 50 - len(obs_step) + i:
                    break
            step = step[i:]
            traj = traj[i:]
            vel = vel[i:]
            agenttype = agenttype[i:]
            psirad = psirad[i:]
            agentcategory = agentcategory[i:]

            feat = np.zeros((110, 2), np.float32)
            feat_vel = np.zeros((110, 2), np.float32)
            feat_agenttype = np.zeros((110, 1), np.float32)
            feat_psirad = np.zeros((110, 1), np.float32)
            feat_agentcategory = np.zeros((110, 2), np.float32)

            # center and rotate positions, rotate velocities
            feat[step] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            feat_vel[step] = np.matmul(rot, vel.T).T

            ### NOTE: max heading is pi, min_heading is -pi (same as INTERACTION)
            # Therefore, heading + theta has min: -pi + 0 = -pi and max: pi + 2pi = 3pi
            for j in range(len(psirad)):
                psirad[j, 0] = psirad[j, 0] + theta
                # angle now between -pi and 2pi
                if psirad[j, 0] >= (2 * math.pi):
                    psirad[j, 0] = psirad[j] % (2 * math.pi)
                # if between pi and 2pi
                if psirad[j, 0] > math.pi:
                    psirad[j, 0] = -1 * ((2 * math.pi) - psirad[j, 0])
            feat_psirad[step] = psirad

            feat_agentcategory[step] = agentcategory
            feat_agenttype[step] = agenttype

            # ctrs contains the centers at the present timestep
            ctrs.append(feat[49, :].copy())

            feat_loc = np.copy(feat)
            # feat contains trajectory offsets
            feat[1:, :] -= feat[:-1, :]
            feat[step[0], :] = 0 

            feats.append(feat)
            feat_locs.append(feat_loc)
            feat_vels.append(feat_vel)
            feat_agenttypes.append(feat_agenttype)
            feat_psirads.append(feat_psirad)
            feat_agentcategories.append(feat_agentcategory)
            gt_preds.append(gt_pred)
            gt_vels.append(gt_vel)
            gt_psirads.append(gt_psirad)
            has_preds.append(has_pred)
            has_obss.append(has_obs)

        ctrs = np.asarray(ctrs, np.float32)
        feats = np.asarray(feats, np.float32)
        feat_locs = np.asarray(feat_locs, np.float32)
        feat_vels = np.asarray(feat_vels, np.float32)
        feat_agenttypes = np.asarray(feat_agenttypes, np.float32)
        feat_psirads = np.asarray(feat_psirads, np.float32)
        feat_agentcategories = np.asarray(feat_agentcategories, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        gt_vels = np.asarray(gt_vels, np.float32)
        gt_psirads = np.asarray(gt_psirads, np.float32)
        has_preds = np.asarray(has_preds, np.float32)
        has_obss = np.asarray(has_obss, np.float32)
        is_valid_agent = np.asarray(is_valid_agent, bool)

        ig_labels_sparse = self.get_interaction_labels_fjmp(idx, ctrs, feat_locs, feat_vels, feat_psirads, has_obss, is_valid_agent, feat_agenttypes, 25)
        ig_labels_sparse = np.asarray(ig_labels_sparse, np.float32)

        ig_labels_dense = self.get_interaction_labels_fjmp(idx, ctrs, feat_locs, feat_vels, feat_psirads, has_obss, is_valid_agent, feat_agenttypes, 60)
        ig_labels_dense = np.asarray(ig_labels_dense, np.float32)

        ig_labels_m2i = self.get_interaction_labels_m2i(idx, ctrs, feat_locs, feat_vels, feat_psirads, has_obss, is_valid_agent, feat_agenttypes)
        ig_labels_m2i = np.asarray(ig_labels_m2i, np.float32)

        # Check that there are no nans
        assert theta <= (2 * math.pi)
        assert theta >= 0
        assert not np.any(np.isnan(ctrs))
        assert not np.any(np.isnan(feats))
        assert not np.any(np.isnan(feat_locs))
        assert not np.any(np.isnan(feat_vels))
        assert not np.any(np.isnan(feat_agenttypes))
        assert not np.any(np.isnan(feat_psirads))
        assert not np.any(np.isnan(feat_agentcategories))
        assert not np.any(np.isnan(gt_preds))
        assert not np.any(np.isnan(gt_vels))
        assert not np.any(np.isnan(has_preds))
        assert not np.any(np.isnan(has_obss))
        assert not np.any(np.isnan(is_valid_agent))
        assert not np.any(np.isnan(ig_labels_sparse))
        assert not np.any(np.isnan(ig_labels_dense))
        assert not np.any(np.isnan(ig_labels_m2i))

        data['feats'] = feats 
        data['ctrs'] = ctrs 
        data['feat_locs'] = feat_locs
        data['feat_vels'] = feat_vels 
        data['feat_agenttypes'] = feat_agenttypes
        data['feat_psirads'] = feat_psirads 
        data['feat_agentcategories'] = feat_agentcategories
        data['gt_preds'] = gt_preds 
        data['gt_vels'] = gt_vels
        data['gt_psirads'] = gt_psirads
        data['has_preds'] = has_preds
        data['has_obss'] = has_obss
        data['orig'] = orig 
        data['theta'] = theta 
        data['rot'] = rot 
        data['is_valid_agent'] = is_valid_agent
        data['ig_labels_sparse'] = ig_labels_sparse
        data['ig_labels_dense'] = ig_labels_dense
        data['ig_labels_m2i'] = ig_labels_m2i

        return data

    def get_interaction_labels_fjmp(self, idx, ctrs, feat_locs, feat_vels, feat_psirads, has_obss, is_valid_agent, feat_agenttypes, eps_I):

        feat_locs = feat_locs[:, 50:]
        feat_vels = feat_vels[:, 50:]
        feat_psirads = feat_psirads[:, 50:]
        
        # only consider the future
        has_obss = has_obss[:, 50:]
        
        N = feat_locs.shape[0]
        labels = np.zeros((N, N))
        orig_trajs = feat_locs 

        circle_lists = []
        for i in range(N):
            length_i = self.avg_agent_length[feat_agenttypes[i, 49, 0]]
            width_i = self.avg_agent_width[feat_agenttypes[i, 49, 0]]
            traj_i = orig_trajs[i][has_obss[i] == 1]
            psirad_i = feat_psirads[i][has_obss[i] == 1]
            # shape is [60, c, 2], where c is the number of circles prescribed to vehicle i (depends on the size/shape of vehicle i)
            circle_lists.append(return_circle_list(traj_i[:, 0], traj_i[:, 1], length_i, width_i, psirad_i[:, 0]))
        
        for a in range(1, N):
            for b in range(a):
                width_a = self.avg_agent_width[feat_agenttypes[a, 49, 0]]
                width_b = self.avg_agent_width[feat_agenttypes[b, 49, 0]]
                # for each (unordered) pairs of vehicles, we check if they are interacting
                # by checking if there is a collision at any pair of future timesteps. 
                circle_list_a = circle_lists[a]
                circle_list_b = circle_lists[b]

                # threshold determined according to widths of vehicles
                thresh = return_collision_threshold(width_a, width_b)

                dist = np.expand_dims(np.expand_dims(circle_list_a, axis=1), axis=2) - np.expand_dims(np.expand_dims(circle_list_b, axis=0), axis=3)
                dist = np.linalg.norm(dist, axis=-1, ord=2)
                
                is_coll = dist < thresh
                is_coll_cumul = is_coll.sum(2).sum(2)

                
                # binary mask of shape [T_a, T_b], where T_a is the number of ground-truth future positions present in a's trajectory, and b defined similarly.
                is_coll_mask = is_coll_cumul > 0

                if is_coll_mask.sum() < 1:
                    continue

                # fill in for indices (0) that do not have a ground-truth position
                for en, ind in enumerate(has_obss[a]):
                    if ind == 0:
                        is_coll_mask = np.insert(is_coll_mask, en, 0, axis=0)

                for en, ind in enumerate(has_obss[b]):
                    if ind == 0:
                        is_coll_mask = np.insert(is_coll_mask, en, 0, axis=1)  

                assert is_coll_mask.shape == (60, 60)

                # [P, 2], first index is a, second is b; P is number of colliding pairs
                coll_ids = np.argwhere(is_coll_mask == 1)
                # only preserve the colliding pairs that are within eps_I (e.g. 6 seconds (= 60 timesteps)) of eachother
                valid_coll_mask = np.abs(coll_ids[:, 0] - coll_ids[:, 1]) <= eps_I

                if valid_coll_mask.sum() < 1:
                    continue

                coll_ids = coll_ids[valid_coll_mask]
                
                # first order small_timestep, larger_timestep, index_of_larger_timestep
                coll_ids_sorted = np.sort(coll_ids, axis=-1)
                coll_ids_argsorted = np.argsort(coll_ids, axis=-1)

                conflict_time_influencer = coll_ids_sorted[:, 0].min()
                influencer_mask = coll_ids_sorted[:, 0] == conflict_time_influencer
                candidate_reactors = coll_ids_sorted[coll_ids_sorted[:, 0] == conflict_time_influencer][:, 1]
                conflict_time_reactor = candidate_reactors.min()
                conflict_time_reactor_id = np.argmin(candidate_reactors)

                a_is_influencer = coll_ids_argsorted[influencer_mask][conflict_time_reactor_id][0] == 0
                if a_is_influencer:
                    min_a = conflict_time_influencer 
                    min_b = conflict_time_reactor 
                else:
                    min_a = conflict_time_reactor 
                    min_b = conflict_time_influencer
                
                # a is the influencer
                if min_a < min_b:
                    labels[a, b] = 1
                # b is the influencer
                elif min_b < min_a:
                    labels[b, a] = 1
                else:                    
                    # if both reach the conflict point at the same timestep, the influencer is the vehicle with the higher velocity @ the conflict point.
                    if np.linalg.norm(feat_vels[a][min_a], ord=2) > np.linalg.norm(feat_vels[b][min_b], ord=2):
                        labels[a, b] = 1
                    elif np.linalg.norm(feat_vels[a][min_a], ord=2) < np.linalg.norm(feat_vels[b][min_b], ord=2):
                        labels[b, a] = 1
                    else:
                        labels[a, b] = 0
                        labels[b, a] = 0
        
        # i --> j iff ig_labels_npy[i,j] = 1
        n_agents = labels.shape[0]

        assert n_agents == np.sum(is_valid_agent)

        # labels for interaction visualization
        valid_mask = is_valid_agent

        # add indices for the invalid agents (either not cars, or no gt position at timestep 9)
        for ind in range(valid_mask.shape[0]):
            if valid_mask[ind] == 0:
                labels = np.insert(labels, ind, 0, axis=1)

        for ind in range(valid_mask.shape[0]):
            if valid_mask[ind] == 0:
                labels = np.insert(labels, ind, 0, axis=0)

        # Here we now construct the interaction labels for SSL.
        # There is a label on each (undirected) edge in the fully connected interaction graph
        ig_labels = np.zeros(int(n_agents * (n_agents - 1) / 2))
        count = 0
        for i in range(len(is_valid_agent)):
            if is_valid_agent[i] == 0:
                assert labels[i].sum() == 0
                continue
            
            for j in range(len(is_valid_agent)):
                if is_valid_agent[j] == 0:
                    assert labels[:,j].sum() == 0
                    continue
                
                # we want only the indices where i < j
                if i >= j:
                    continue 

                if labels[i, j] == 1:
                    # i influences j
                    ig_labels[count] = 1
                    # j influences i
                elif labels[j, i] == 1:
                    ig_labels[count] = 2
                
                count += 1   

        assert ig_labels.shape[0] == count

        return ig_labels

    def get_interaction_labels_m2i(self, idx, ctrs, feat_locs, feat_vels, feat_psirads, has_obss, is_valid_agent, feat_agenttypes):
        """
        feat_locs: location features in transformed coordinates (not offsets but absolute positions) (past + future): [N, 40, 2]
        feat_vels: velocity features (past + future): [N, 40, 2]
        shapes: vehicle shape: [N, 40, 2] (length, width)
        has_obss: ground-truth mask (past + future): [N, 40]
        is_valid_agent: whether the agent is being considered during training (only cars considered): [N, ]
        """
        
        N = feat_locs.shape[0]
        # NOTE: labels[i, j] = 0 if no interaction exists, = 1 if i --> j, = 2 if j --> i
        labels = np.zeros((N, N))

        orig_trajs = feat_locs
        for a in range(1, N):
            for b in range(a):
                # sum of the length of these two vehicles.               
                len_a = self.avg_agent_length[feat_agenttypes[a, 49, 0]]
                if np.isnan(len_a):
                    print("This should not happen")
                    len_a = 1
                len_b =  self.avg_agent_length[feat_agenttypes[b, 49, 0]]
                if np.isnan(len_b):
                    print("This should not happen")
                    len_b = 1
                
                EPSILON_D = len_a + len_b
                
                # filter for the timesteps with a ground-truth position
                traj_a = orig_trajs[a][has_obss[a] == 1]
                traj_b = orig_trajs[b][has_obss[b] == 1]

                traj_a_expanded = traj_a.reshape(-1, 1, 2)
                traj_b_expanded = traj_b.reshape(1, -1, 2)

                # [A, B] array, where A = traj_a.shape[0], B = traj_a.shape[1]
                dist_ab = np.sqrt(np.sum((traj_a_expanded - traj_b_expanded)**2, axis=2))

                # fill in for indices that do not have a ground-truth position
                for en, ind in enumerate(has_obss[a]):
                    if ind == 0:
                        dist_ab = np.insert(dist_ab, en, 10000, axis=0)

                for en, ind in enumerate(has_obss[b]):
                    if ind == 0:
                        dist_ab = np.insert(dist_ab, en, 10000, axis=1)   

                # broadcast back into a length 110 tensor first.
                assert dist_ab.shape == (110, 110) 

                # We only consider the future positions, as the past positions are already fed into the model.
                dist_ab = dist_ab[50:, 50:]            

                # in [0, 59] (future timestep)
                min_a, min_b = np.unravel_index(dist_ab.argmin(), dist_ab.shape)
                
                if np.min(dist_ab) > EPSILON_D:
                    continue 
                
                if min_a < min_b:
                    labels[a, b] = 1
                elif min_b < min_a:
                    labels[b, a] = 1
                else:                    
                    # if both reach the conflict point at the same timestep, the influencer is the vehicle with the higher velocity @ the conflict point.
                    if np.linalg.norm(feat_vels[a][min_a + 50], ord=2) > np.linalg.norm(feat_vels[b][min_b + 50], ord=2):
                        labels[a, b] = 1
                    elif np.linalg.norm(feat_vels[a][min_a + 50], ord=2) < np.linalg.norm(feat_vels[b][min_b + 50], ord=2):
                        labels[b, a] = 1
                    else:
                        labels[a, b] = 0
                        labels[b, a] = 0

        # i --> j iff ig_labels_npy[i,j] = 1
        n_agents = labels.shape[0]

        assert n_agents == np.sum(is_valid_agent)

        # labels for interaction visualization
        valid_mask = is_valid_agent

        # add indices for the invalid agents (no gt position at timestep 49)
        for ind in range(valid_mask.shape[0]):
            if valid_mask[ind] == 0:
                labels = np.insert(labels, ind, 0, axis=1)

        for ind in range(valid_mask.shape[0]):
            if valid_mask[ind] == 0:
                labels = np.insert(labels, ind, 0, axis=0)

        # Here we now construct the interaction labels for SSL.
        # There is a label on each (undirected) edge in the fully connected interaction graph
        ig_labels = np.zeros(int(n_agents * (n_agents - 1) / 2))
        count = 0
        for i in range(len(is_valid_agent)):
            if is_valid_agent[i] == 0:
                assert labels[i].sum() == 0
                continue
            
            for j in range(len(is_valid_agent)):
                if is_valid_agent[j] == 0:
                    assert labels[:,j].sum() == 0
                    continue
                
                # we want only the indices where i < j
                if i >= j:
                    continue 

                if labels[i, j] == 1:
                    # i influences j
                    ig_labels[count] = 1
                    # j influences i
                elif labels[j, i] == 1:
                    ig_labels[count] = 2
                
                count += 1   

        assert ig_labels.shape[0] == count

        return ig_labels

    def get_lane_graph(self, data, idx):
        scene_directory = self.mapping[idx]
        static_map_path = os.path.join(self.files, scene_directory, "log_map_archive_{}.json".format(scene_directory))
        static_map = ArgoverseStaticMap.from_json(Path(static_map_path))

        lane_ids, ctrs, feats = [], [], []
        centerlines, left_boundaries, right_boundaries = [], [], []
        for lane_segment in static_map.vector_lane_segments.values():
            left_boundary = copy.deepcopy(lane_segment.left_lane_boundary.xyz[:, :2])
            right_boundary = copy.deepcopy(lane_segment.right_lane_boundary.xyz[:, :2])
            centerline, _ = compute_midpoint_line(left_boundary, right_boundary, min(10, max(left_boundary.shape[0], right_boundary.shape[0])))
            centerline = copy.deepcopy(centerline)             
            
            # process lane centerline in same way as agent trajectories
            centerline = np.matmul(data['rot'], (centerline - data['orig'].reshape(-1, 2)).T).T
            left_boundary = np.matmul(data['rot'], (left_boundary - data['orig'].reshape(-1, 2)).T).T
            right_boundary = np.matmul(data['rot'], (right_boundary - data['orig'].reshape(-1, 2)).T).T
        
            num_segs = len(centerline) - 1
            # locations between the centerline segments
            ctrs.append(np.asarray((centerline[:-1] + centerline[1:]) / 2.0, np.float32))
            # centerline segment offsets
            feats.append(np.asarray(centerline[1:] - centerline[:-1], np.float32))
            lane_ids.append(lane_segment.id)
            centerlines.append(centerline)
            left_boundaries.append(left_boundary)
            right_boundaries.append(right_boundary)

        # node indices (when nodes are concatenated into one array)
        node_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count

        # predecessors and successors of a lane
        pre, suc = dict(), dict()
        for key in ['u', 'v']:
            pre[key], suc[key] = [], []

        for i, lane_segment in enumerate(static_map.vector_lane_segments.values()):
            idcs = node_idcs[i]

            # points to the predecessor
            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]
            if lane_segment.predecessors is not None:
                for nbr_id in lane_segment.predecessors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])

            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]
            if lane_segment.successors is not None:
                for nbr_id in lane_segment.successors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])
        
        # we now compute lane-level features
        # lane indices
        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)

        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_segment in enumerate(static_map.vector_lane_segments.values()):
            lane = lane_segment 

            nbr_ids = lane.predecessors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre_pairs.append([i, j])

            nbr_ids = lane.successors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc_pairs.append([i, j])

            nbr_id = lane.left_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    left_pairs.append([i, j])

            nbr_id = lane.right_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    right_pairs.append([i, j])
        
        pre_pairs = np.asarray(pre_pairs, np.int64)
        suc_pairs = np.asarray(suc_pairs, np.int64)
        left_pairs = np.asarray(left_pairs, np.int64)
        right_pairs = np.asarray(right_pairs, np.int64)

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['centerlines'] = centerlines
        graph['left_boundaries'] = left_boundaries
        graph['right_boundaries'] = right_boundaries
        graph['pre'] = [pre]
        graph['suc'] = [suc]
        graph['lane_idcs'] = lane_idcs
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs

        for k1 in ['pre', 'suc']:
            for k2 in ['u', 'v']:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)
        
        # longitudinal connections
        for key in ['pre', 'suc']:
            graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], self.config['num_scales'])

        return graph

