import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
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
import lanelet2
from lanelet2.projection import UtmProjector
from av2.geometry.interpolate import compute_midpoint_line

np.set_printoptions(suppress=True)

class InteractionDataset(Dataset):
    def __init__(self, config, train=True, train_all=False):
        self.config = config
        self.train = train

        if self.train:
            self.filename_pattern = re.compile(r'^(\w+)_train_\d+.csv$')
            self.mapping_filename = 'mapping_train.pkl'
            if train_all:
                self.n_samples = 47584 + 11794
            else:
                self.n_samples = 47584
            self.tracks = os.path.join(self.config['dataset_path'], 'train')
            self.tracks_reformatted = self.config['tracks_train_reformatted']
            self.preprocess_path = self.config["preprocess_train"]
        else:
            self.filename_pattern = re.compile(r'^(\w+)_val_\d+.csv$')
            self.mapping_filename = 'mapping_val.pkl'
            self.n_samples = 11794
            self.tracks = os.path.join(self.config['dataset_path'], 'val')
            self.tracks_reformatted = self.config['tracks_val_reformatted']
            self.preprocess_path = self.config["preprocess_val"]

        if not os.path.isdir(self.tracks_reformatted):
            os.makedirs(self.tracks_reformatted)   

        self.projector = UtmProjector(lanelet2.io.Origin(0, 0))
        self.traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                            lanelet2.traffic_rules.Participants.Vehicle)  

        # first check if mapping file exists         
        if not os.path.isfile(os.path.join(self.config['dataset_path'], self.mapping_filename)):
            print("Reformatting dataset...")

            mapping = {}
            idx = 0
            for csv_file in os.listdir(self.tracks):
                if not csv_file.endswith('.csv'):
                    continue

                csv_path = os.path.join(self.tracks, csv_file)
                with open(csv_path, "r") as src:
                    rdr = csv.reader(src)
                    rdr = sorted(rdr, key = lambda r: r[0], reverse=True)

                    scenarios = {}

                    for r in rdr:
                        
                        if r[0] == "case_id" and r[0] not in scenarios:
                            scenarios["header"] = self.row(r)
                            continue
                        
                        if r[0] not in scenarios:
                            scenarios[r[0]] = [self.row(r)]
                        else:
                            scenarios[r[0]].append(self.row(r))

                    for scenario in scenarios:
                        if scenario == "header":
                            continue
                        
                        track_reformatted = csv_file[:-4] + "_{}".format(int(float(scenario))) + ".csv"
                        track_reformatted_path = os.path.join(self.tracks_reformatted, track_reformatted)
                        with open(track_reformatted_path, "w") as res:
                            wtr = csv.writer(res)
                            wtr.writerow(scenarios["header"])
                            for r in scenarios[scenario]:
                                wtr.writerow(r)

                        mapping[idx] = track_reformatted
                        idx += 1
                
            with open(os.path.join(self.config['dataset_path'], self.mapping_filename), "wb") as f:
                pickle.dump(mapping, f)

        # load mapping dictionary
        with open(os.path.join(self.config['dataset_path'], self.mapping_filename), "rb") as f:
            self.mapping = pickle.load(f)

        self.avg_pedcyc_length = 0.7
        self.avg_pedcyc_width = 0.7
    
    def row(self, r):
        return (r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11])
    
    def __len__(self):
        return self.n_samples

    def read_interaction_data(self, idx):
        csv_file = self.mapping[idx]
        city = self.filename_pattern.match(csv_file).group(1)
        csv_path = os.path.join(self.tracks_reformatted, csv_file)

        """TRACK_ID,FRAME_ID,TIMESTAMP_MS,AGENT_TYPE,X,Y,VX,VY,PSI_RAD,LENGTH,WIDTH"""
        df = read_csv(csv_path)

        agt_ts = np.sort(np.unique(df['timestamp_ms'].values))
        timestamp_mapping = dict()
        for i, ts in enumerate(agt_ts):
            timestamp_mapping[ts] = i

        trajs = np.concatenate((
            df.x.to_numpy().reshape(-1, 1),
            df.y.to_numpy().reshape(-1, 1)
        ), 1)

        vels = np.concatenate((
            df.vx.to_numpy().reshape(-1, 1),
            df.vy.to_numpy().reshape(-1, 1)
        ), 1)

        psirads = df.psi_rad.to_numpy().reshape(-1, 1)

        agenttypes = df.agent_type
        agenttypes = np.array([1 if x == 'car' else 0 for x in agenttypes]).reshape(-1, 1)

        shapes = np.concatenate((
            df.length.to_numpy().reshape(-1, 1),
            df.width.to_numpy().reshape(-1, 1)
        ), 1)

        # the timestep indices the trajectory contains
        steps = [timestamp_mapping[x] for x in df['timestamp_ms'].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['track_id']).groups 
        keys = list(objs.keys())
        ctx_trajs, ctx_steps, ctx_vels, ctx_psirads, ctx_shapes, ctx_agenttypes = [], [], [], [], [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])
            ctx_vels.append(vels[idcs])
            ctx_psirads.append(psirads[idcs])
            ctx_shapes.append(shapes[idcs])
            ctx_agenttypes.append(agenttypes[idcs])           

        data = dict()
        data['city'] = city 
        data['trajs'] = ctx_trajs
        data['steps'] = ctx_steps 
        data['vels'] = ctx_vels
        data['psirads'] = ctx_psirads
        data['shapes'] = ctx_shapes
        data['agenttypes'] = ctx_agenttypes

        return data

    def get_interaction_labels_sparse(self, idx, ctrs, feat_locs, feat_vels, feat_psirads, shapes, has_obss, is_valid_agent, agenttypes):

        # only consider the future
        # we can use data in se(2) transformed coordinates (interaction labelling invariant to se(2)-transformations)
        feat_locs = feat_locs[:, 10:]
        feat_vels = feat_vels[:, 10:]
        feat_psirads = feat_psirads[:, 10:]
        has_obss = has_obss[:, 10:]
        
        N = feat_locs.shape[0]
        labels = np.zeros((N, N))
        orig_trajs = feat_locs 

        circle_lists = []
        for i in range(N):
            agenttype_i = agenttypes[i][9]
            if agenttype_i == 1:
                shape_i = shapes[i][9]
                length = shape_i[0]
                width = shape_i[1]
            else:
                length = self.avg_pedcyc_length
                width = self.avg_pedcyc_width

            traj_i = orig_trajs[i][has_obss[i] == 1]
            psirad_i = feat_psirads[i][has_obss[i] == 1]
            # shape is [30, c, 2], where c is the number of circles prescribed to vehicle i (depends on the size/shape of vehicle i)
            circle_lists.append(return_circle_list(traj_i[:, 0], traj_i[:, 1], length, width, psirad_i[:, 0]))
        
        for a in range(1, N):
            for b in range(a):
                agenttype_a = agenttypes[a][9]
                if agenttype_a == 1:
                    shape_a = shapes[a][9]
                    width_a = shape_a[1]
                else:
                    width_a = self.avg_pedcyc_width

                agenttype_b = agenttypes[b][9]
                if agenttype_b == 1:
                    shape_b = shapes[b][9]
                    width_b = shape_b[1]
                else:
                    width_b = self.avg_pedcyc_width
                
                # for each (unordered) pairs of vehicles, we check if they are interacting
                # by checking if there is a collision at any pair of future timesteps. 
                circle_list_a = circle_lists[a]
                circle_list_b = circle_lists[b]

                # threshold determined according to widths of vehicles
                thresh = return_collision_threshold(width_a, width_b)

                dist = np.expand_dims(np.expand_dims(circle_list_a, axis=1), axis=2) - np.expand_dims(np.expand_dims(circle_list_b, axis=0), axis=3)
                dist = np.linalg.norm(dist, axis=-1, ord=2)
                
                # [T_a, T_b, num_circles_a, num_circles_b], where T_a is the number of ground-truth future positions present in a's trajectory, and b defined similarly.
                is_coll = dist < thresh
                is_coll_cumul = is_coll.sum(2).sum(2)
                # binary mask of shape [T_a, T_b]
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

                assert is_coll_mask.shape == (30, 30)

                # [P, 2], first index is a, second is b; P is number of colliding pairs
                coll_ids = np.argwhere(is_coll_mask == 1)
                # only preserve the colliding pairs that are within 2.5 seconds (= 25 timesteps) of eachother
                valid_coll_mask = np.abs(coll_ids[:, 0] - coll_ids[:, 1]) <= 25

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
        
        n_agents = labels.shape[0]

        assert n_agents == np.sum(is_valid_agent)

        # labels for interaction visualization
        valid_mask = is_valid_agent

        # add indices for the invalid agents (no gt position at timestep 9)
        for ind in range(valid_mask.shape[0]):
            if valid_mask[ind] == 0:
                labels = np.insert(labels, ind, 0, axis=1)

        for ind in range(valid_mask.shape[0]):
            if valid_mask[ind] == 0:
                labels = np.insert(labels, ind, 0, axis=0)

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

                # i influences j
                if labels[i, j] == 1:
                    ig_labels[count] = 1
                # j influences i
                elif labels[j, i] == 1:
                    ig_labels[count] = 2
                
                count += 1   

        assert ig_labels.shape[0] == count

        return ig_labels
    
    def get_interaction_labels_dense(self, idx, ctrs, feat_locs, feat_vels, shapes, has_obss, is_valid_agent, agenttypes):
        
        N = feat_locs.shape[0]
        # labels[i, j] = 0 if no interaction exists, = 1 if i --> j, = 2 if j --> i
        labels = np.zeros((N, N))

        orig_trajs = feat_locs
        for a in range(1, N):
            for b in range(a):
                agenttype_a = agenttypes[a][9]
                if agenttype_a == 1:
                    shape_a = shapes[a][9]
                    len_a = shape_a[0]
                else:
                    len_a = self.avg_pedcyc_length

                agenttype_b = agenttypes[b][9]
                if agenttype_b == 1:
                    shape_b = shapes[b][9]
                    len_b = shape_b[0]
                else:
                    len_b = self.avg_pedcyc_length
                
                # sum of the lengths of the two agents
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

                # broadcast back into a length 40 tensor first.
                assert dist_ab.shape == (40, 40) 

                # We only consider the future positions, as the past positions are already fed into the model.
                dist_ab = dist_ab[10:, 10:]            

                # in [0, 29] (future timestep)
                min_a, min_b = np.unravel_index(dist_ab.argmin(), dist_ab.shape)
                
                if np.min(dist_ab) > EPSILON_D:
                    continue 
                
                if min_a < min_b:
                    labels[a, b] = 1
                elif min_b < min_a:
                    labels[b, a] = 1
                else:                    
                    # if both reach the conflict point at the same timestep, the influencer is the vehicle with the higher velocity @ the conflict point.
                    if np.linalg.norm(feat_vels[a][min_a + 10], ord=2) > np.linalg.norm(feat_vels[b][min_b + 10], ord=2):
                        labels[a, b] = 1
                    elif np.linalg.norm(feat_vels[a][min_a + 10], ord=2) < np.linalg.norm(feat_vels[b][min_b + 10], ord=2):
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

                # i influences j
                if labels[i, j] == 1:
                    ig_labels[count] = 1
                # j influences i
                elif labels[j, i] == 1:
                    ig_labels[count] = 2
                
                count += 1   

        assert ig_labels.shape[0] == count

        return ig_labels


    def get_obj_feats(self, data, idx):
        # center on "random" agent
        # This ensures that the random agent chosen is same for loaded graph
        # processed on lanelet2-compatible machine and agent processed on other machine.
        if self.train:
            orig_idx =  idx % len(data['trajs'])
            while True:
                # Is the present timestep available for this agent?
                if 9 in data['steps'][orig_idx]:
                    break 
                else:
                    orig_idx = (orig_idx + 1) % len(data['trajs'])

        # center on the agent closest to the centroid
        else:
            scored_ctrs = []
            scored_indices = []
            for i in range(len(data['trajs'])):
                if 8 in data['steps'][i] and 9 in data['steps'][i] and 39 in data['steps'][i]:
                    present_index = list(data['steps'][i]).index(9)
                    scored_ctrs.append(np.expand_dims(data['trajs'][i][present_index], axis=0))
                    scored_indices.append(i)

            scored_ctrs = np.concatenate(scored_ctrs, axis=0)
            centroid = np.mean(scored_ctrs, axis=0)
            dist_to_centroid = np.linalg.norm(scored_ctrs - np.expand_dims(centroid, axis=0), ord=2, axis=-1)
            closest_centroid = np.argmin(dist_to_centroid, axis=0)
            orig_idx = scored_indices[closest_centroid]
        
        orig = data['trajs'][orig_idx][9].copy().astype(np.float32)
        pre = data['trajs'][orig_idx][8] - orig 
        theta = np.pi - np.arctan2(pre[1], pre[0])

        # rotation matrix for rotating scene
        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)
        
        feats, feat_locs, feat_vels, gt_preds, gt_vels, has_preds, has_obss = [], [], [], [], [], [], []
        feat_psirads, feat_shapes, gt_psirads, ctrs, feat_agenttypes = [], [], [], [], []
        is_valid_agent = []
        
        for traj, step, vel, psirad, shape, agenttype in zip(data['trajs'], data['steps'], data['vels'], data['psirads'], data['shapes'], data['agenttypes']):
            if 9 not in step:
                is_valid_agent.append(0)
                continue
            
            is_valid_agent.append(1)
            
            # ground-truth future positions
            gt_pred = np.zeros((30, 2), np.float32)
            # ground truth future velocities
            gt_vel = np.zeros((30, 2), np.float32)
            # ground truth yaw angles
            gt_psirad = np.zeros((30, 1), np.float32)
            
            # has ground-truth future mask
            has_pred = np.zeros(30, bool)
            has_obs = np.zeros(40, bool)

            future_mask = np.logical_and(step >= 10, step < 40)
            post_step = step[future_mask] - 10
            post_traj = traj[future_mask]
            post_vel = vel[future_mask]
            post_agenttype = agenttype[future_mask]
            post_psirad = np.nan_to_num(psirad[future_mask]) * post_agenttype # 0 out psirad if not a car
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
            shape = shape[idcs]
            has_obs[step] = 1

            # only observation horizon
            obs_step = step[step < 10]
            obs_idcs = obs_step.argsort()
            obs_step = obs_step[obs_idcs]

            # take contiguous past to be the past
            for i in range(len(obs_step)):
                if obs_step[i] == 10 - len(obs_step) + i:
                    break
            step = step[i:]
            traj = traj[i:]
            vel = vel[i:]
            agenttype = agenttype[i:]
            psirad = psirad[i:]
            shape = shape[i:]

            feat = np.zeros((40, 2), np.float32)
            feat_vel = np.zeros((40, 2), np.float32)
            feat_agenttype = np.zeros((40, 1), np.float32)
            feat_psirad = np.zeros((40, 1), np.float32)
            feat_shape = np.zeros((40, 2), np.float32)
            
            # center and rotate positions, rotate velocities
            feat[step] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            feat_vel[step] = np.matmul(rot, vel.T).T

            # recalculate yaw angles
            feat_agenttype[step] = agenttype
            feat_shape[step] = np.nan_to_num(shape) * feat_agenttype[step] # 0 out if not a car 
            
            # only vehicles have a yaw angle
            # apply rotation transformation to the yaw angle
            if feat_agenttype[9] != 0:
                for j in range(len(psirad)):
                    psirad[j, 0] = psirad[j, 0] + theta
                    # angle now between -pi and 2pi
                    if psirad[j, 0] >= (2 * math.pi):
                        psirad[j, 0] = psirad[j] % (2 * math.pi)
                    # if between pi and 2pi
                    if psirad[j, 0] > math.pi:
                        psirad[j, 0] = -1 * ((2 * math.pi) - psirad[j, 0])
            # pedestrian/bicycle does not have yaw angle; use velocity to infer yaw when available; otherwise set to 0
            # velocity is already in se(2) transformed coordinates
            else:
                vel_transformed = feat_vel[step]
                assert len(psirad) == len(vel_transformed)
                for j in range(len(psirad)):
                    speed_j = math.sqrt(vel_transformed[j, 0] ** 2 + vel_transformed[j, 1] ** 2)
                    if speed_j == 0:
                        psirad[j, 0] = 0.
                    else:
                        psirad[j, 0] = round(sign_func(vel_transformed[j, 1]) * math.acos(vel_transformed[j, 0] / speed_j), 3)

            assert not np.any(np.isnan(psirad))
            feat_psirad[step] = psirad
            
            # ctrs contains the centers at the present timestep
            ctrs.append(feat[9, :].copy())
            
            feat_loc = np.copy(feat)
            # feat contains trajectory offsets
            feat[1:, :] -= feat[:-1, :]
            feat[step[0], :] = 0 

            feats.append(feat)
            feat_locs.append(feat_loc)
            feat_vels.append(feat_vel)
            feat_agenttypes.append(feat_agenttype)
            feat_psirads.append(feat_psirad)
            feat_shapes.append(feat_shape)
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
        feat_shapes = np.asarray(feat_shapes, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        gt_vels = np.asarray(gt_vels, np.float32)
        gt_psirads = np.asarray(gt_psirads, np.float32)
        has_preds = np.asarray(has_preds, np.float32)
        has_obss = np.asarray(has_obss, np.float32)
        is_valid_agent = np.asarray(is_valid_agent, bool)

        ig_labels_dense = self.get_interaction_labels_dense(idx, ctrs, feat_locs, feat_vels, feat_shapes, has_obss, is_valid_agent, feat_agenttypes)
        ig_labels_dense = np.asarray(ig_labels_dense, np.float32)

        ig_labels_sparse = self.get_interaction_labels_sparse(idx, ctrs, feat_locs, feat_vels, feat_psirads, feat_shapes, has_obss, is_valid_agent, feat_agenttypes)
        ig_labels_sparse = np.asarray(ig_labels_sparse, np.float32)

        # Check that there are no nans
        assert theta <= (2 * math.pi)
        assert theta >= 0
        assert not np.any(np.isnan(ctrs))
        assert not np.any(np.isnan(feats))
        assert not np.any(np.isnan(feat_locs))
        assert not np.any(np.isnan(feat_vels))
        assert not np.any(np.isnan(feat_agenttypes))
        assert not np.any(np.isnan(feat_psirads))
        assert not np.any(np.isnan(feat_shapes))
        assert not np.any(np.isnan(gt_preds))
        assert not np.any(np.isnan(gt_vels))
        assert not np.any(np.isnan(has_preds))
        assert not np.any(np.isnan(has_obss))
        assert not np.any(np.isnan(is_valid_agent))
        assert not np.any(np.isnan(ig_labels_dense))
        assert not np.any(np.isnan(ig_labels_sparse))

        data['feats'] = feats 
        data['ctrs'] = ctrs 
        data['feat_locs'] = feat_locs
        data['feat_vels'] = feat_vels 
        data['feat_agenttypes'] = feat_agenttypes
        data['feat_psirads'] = feat_psirads 
        data['feat_shapes'] = feat_shapes
        data['gt_preds'] = gt_preds 
        data['gt_vels'] = gt_vels
        data['gt_psirads'] = gt_psirads
        data['has_preds'] = has_preds
        data['has_obss'] = has_obss
        data['orig'] = orig 
        data['theta'] = theta 
        data['rot'] = rot 
        data['is_valid_agent'] = is_valid_agent
        data['ig_labels_dense'] = ig_labels_dense
        data['ig_labels_sparse'] = ig_labels_sparse

        return data

    def get_lane_graph(self, data):
        # Note that we process the full lane graph
        map_path = os.path.join(self.config['maps'], data['city'] + '.osm')
        map = lanelet2.io.load(map_path, self.projector)
        routing_graph = lanelet2.routing.RoutingGraph(map, self.traffic_rules)

        # build node features
        lane_ids, ctrs, feats = [], [], []
        centerlines, left_boundaries, right_boundaries = [], [], []
        for ll in map.laneletLayer:
            left_boundary = np.zeros((len(ll.leftBound), 2))
            right_boundary  = np.zeros((len(ll.rightBound), 2))

            for i in range(len(ll.leftBound)):
                left_boundary[i][0] = copy.deepcopy(ll.leftBound[i].x)
                left_boundary[i][1] = copy.deepcopy(ll.leftBound[i].y)

            for i in range(len(ll.rightBound)):
                right_boundary[i][0] = copy.deepcopy(ll.rightBound[i].x)
                right_boundary[i][1] = copy.deepcopy(ll.rightBound[i].y)
            
            # computes centerline with min(max(M,N), 10) data points per lanelet
            centerline, _ = compute_midpoint_line(left_boundary, right_boundary, min(10, max(left_boundary.shape[0], right_boundary.shape[0])))
            centerline = copy.deepcopy(centerline)            

            # process lane centerline in same way as agent trajectories
            centerline = np.matmul(data['rot'], (centerline - data['orig'].reshape(-1, 2)).T).T
            left_boundary = np.matmul(data['rot'], (left_boundary - data['orig'].reshape(-1, 2)).T).T
            right_boundary = np.matmul(data['rot'], (right_boundary - data['orig'].reshape(-1, 2)).T).T
            
            num_segs = len(centerline) - 1
            # locations between the centerline segments
            ctrs.append(np.asarray((centerline[:-1] + centerline[1:]) / 2.0, np.float32))
            # offsets between the centerline segments
            feats.append(np.asarray(centerline[1:] - centerline[:-1], np.float32))
            lane_ids.append(ll.id)
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
        
        for i, lane_id in enumerate(lane_ids):
            lane = map.laneletLayer[lane_id]
            idcs = node_idcs[i]

            # points to the predecessor
            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]
            if len(routing_graph.previous(lane)) > 0:
                for prev_lane in routing_graph.previous(lane):
                    if prev_lane.id in lane_ids:
                        j = lane_ids.index(prev_lane.id)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])

            # points to the successor
            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]
            if len(routing_graph.following(lane)) > 0:
                for foll_lane in routing_graph.following(lane):
                    if foll_lane.id in lane_ids:
                        j = lane_ids.index(foll_lane.id)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])

        # we now compute lane-level features
        # lane indices
        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)

        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = map.laneletLayer[lane_id]

            # compute lane_id pairs of predecessor [u,v]
            if len(routing_graph.previous(lane)) > 0:
                for prev_lane in routing_graph.previous(lane):
                    if prev_lane.id in lane_ids:
                        j = lane_ids.index(prev_lane.id)
                        pre_pairs.append([i, j])

            # compute lane_id pairs of successor [u,v]
            if len(routing_graph.following(lane)) > 0:
                for foll_lane in routing_graph.following(lane):
                    if foll_lane.id in lane_ids:
                        j = lane_ids.index(foll_lane.id)
                        suc_pairs.append([i, j])

            # compute lane_id pairs of left [u,v]
            if routing_graph.left(lane) is not None:
                if routing_graph.left(lane).id in lane_ids:
                    j = lane_ids.index(routing_graph.left(lane).id)
                    left_pairs.append([i, j])

            # compute lane_id pairs of right [u,v]
            if routing_graph.right(lane) is not None:
                if routing_graph.right(lane).id in lane_ids:
                    j = lane_ids.index(routing_graph.right(lane).id)
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

    def __getitem__(self, idx):
        if self.config['preprocess']:
            data = np.load(os.path.join(self.preprocess_path, "{}.p".format(idx)), allow_pickle=True)

            new_data = dict()
            for key in ['idx', # unique integer assigned to each scene (assigned when building mapping dictionary)
                        'orig', # origin of scene (taken to be present location of random vehicle in scene during training and of centroid vehicle during validation)
                        'feats', # position features (offsets) in SE(2)-transformed coordinate system (past + future)
                        'feat_locs', # location features (not offsets but absolute positions in SE(2)-transformed coordinate system) (past + future)
                        'feat_vels', # velocity features, in SE(2)-transformed coordinate system (past + future)
                        'feat_agenttypes', # agent_type (either car or pedestrian/bicycle, past + future)
                        'feat_psirads', # yaw angle features, in SE(2)-transformed coordinate system (past + future)
                        'feat_shapes', # vehicle [length,width] (past + future)
                        'gt_preds', # ground-truth positions (future)
                        'gt_vels', # ground-truth velocities (future)
                        'gt_psirads', # ground-truth yaw angles (future)
                        'has_preds', # future timestep exists mask (future)
                        'has_obss', # past timestep exists mask (past+future)
                        'theta', # angle for rotating scene
                        'rot', # rotation matrix for rotating scene
                        'ctrs', # agent centers at the present timestep in SE(2)-transformed coordinate system
                        'ig_labels_sparse', # interaction graph labels for current scene (fjmp interaction heuristic)
                        'ig_labels_dense', # interaction graph labels for current scene (m2i interaction heuristic)
                        'graph']: # lane graph
                if key in data:
                    new_data[key] = ref_copy(data[key])
            data = new_data

            return data

        data = self.read_interaction_data(idx)
        data = self.get_obj_feats(data, idx)
        data['idx'] = idx
        data['graph'] = self.get_lane_graph(data)

        return data

class InteractionTestDataset(Dataset):
    def __init__(self, config):
        self.config = config 
        self.filename_pattern = re.compile(r'^(\w+)_obs_(\d+).csv$')
        self.mapping_filename = 'mapping_test.pkl'
        self.n_samples = 2644
        self.tracks = os.path.join(self.config['dataset_path'], 'test_multi-agent')
        self.tracks_reformatted = self.config['tracks_test_reformatted']
        self.preprocess_path = self.config["preprocess_test"]

        if not os.path.isdir(self.tracks_reformatted):
            os.makedirs(self.tracks_reformatted)

        self.projector = UtmProjector(lanelet2.io.Origin(0, 0))
        self.traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                            lanelet2.traffic_rules.Participants.Vehicle)

        # first check if mapping file exists         
        if not os.path.isfile(os.path.join(self.config['dataset_path'], self.mapping_filename)):
            print("Reformatting dataset...")

            mapping = {}
            idx = 0
            for csv_file in os.listdir(self.tracks):
                if not csv_file.endswith('.csv'):
                    continue

                csv_path = os.path.join(self.tracks, csv_file)
                with open(csv_path, "r") as src:
                    rdr = csv.reader(src)
                    rdr = sorted(rdr, key = lambda r: r[0], reverse=True)

                    scenarios = {}

                    for r in rdr:
                        
                        if r[0] == "case_id" and r[0] not in scenarios:
                            scenarios["header"] = self.row(r)
                            continue
                        
                        if r[0] not in scenarios:
                            scenarios[r[0]] = [self.row(r)]
                        else:
                            scenarios[r[0]].append(self.row(r))

                    for scenario in scenarios:
                        if scenario == "header":
                            continue
                        
                        track_reformatted = csv_file[:-4] + "_{}".format(int(float(scenario))) + ".csv"
                        track_reformatted_path = os.path.join(self.tracks_reformatted, track_reformatted)
                        with open(track_reformatted_path, "w") as res:
                            wtr = csv.writer(res)
                            wtr.writerow(scenarios["header"])
                            for r in scenarios[scenario]:
                                wtr.writerow(r)

                        mapping[idx] = track_reformatted
                        idx += 1
                
            with open(os.path.join(self.config['dataset_path'], self.mapping_filename), "wb") as f:
                pickle.dump(mapping, f)
        
        # load mapping dictionary
        with open(os.path.join(self.config['dataset_path'], self.mapping_filename), "rb") as f:
            self.mapping = pickle.load(f)             


    def __len__(self):
        return self.n_samples 

    def row(self, r):
        return (r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13])

    def read_interaction_data(self, idx):
        csv_file = self.mapping[idx]    
        city = self.filename_pattern.match(csv_file).group(1)
        csv_path = os.path.join(self.tracks_reformatted, csv_file)

        """TRACK_ID,FRAME_ID,TIMESTAMP_MS,AGENT_TYPE,X,Y,VX,VY,PSI_RAD,LENGTH,WIDTH,TRACK_TO_PREDICT,INTERESTING_AGENT"""
        df = read_csv(csv_path)

        agt_ts = np.sort(np.unique(df['timestamp_ms'].values))
        timestamp_mapping = dict()
        for i, ts in enumerate(agt_ts):
            timestamp_mapping[ts] = i

        trajs = np.concatenate((
            df.x.to_numpy().reshape(-1, 1),
            df.y.to_numpy().reshape(-1, 1)
        ), 1)

        vels = np.concatenate((
            df.vx.to_numpy().reshape(-1, 1),
            df.vy.to_numpy().reshape(-1, 1)
        ), 1)

        psirads = df.psi_rad.to_numpy().reshape(-1, 1)

        track_ids = df.track_id.to_numpy().reshape(-1, 1)

        agent_to_predicts = df.track_to_predict.to_numpy().reshape(-1, 1)

        interesting_agents = df.interesting_agent.to_numpy().reshape(-1, 1)

        agenttypes = df.agent_type
        agenttypes = np.array([1 if x == 'car' else 0 for x in agenttypes]).reshape(-1, 1)

        shapes = np.concatenate((
            df.length.to_numpy().reshape(-1, 1),
            df.width.to_numpy().reshape(-1, 1)
        ), 1)

        steps = [timestamp_mapping[x] for x in df['timestamp_ms'].values]
        steps = np.asarray(steps, np.int64)

        # We don't group by agent_type as we predict futures of all agents in the scene
        objs = df.groupby(['track_id']).groups 
        keys = list(objs.keys())
        ctx_trajs, ctx_steps, ctx_vels, ctx_psirads, ctx_track_ids, ctx_agent_to_predicts, ctx_interesting_agents, ctx_shapes, ctx_agenttypes = [], [], [], [], [], [], [], [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])
            ctx_vels.append(vels[idcs])
            ctx_psirads.append(psirads[idcs])
            ctx_track_ids.append(track_ids[idcs])
            ctx_agent_to_predicts.append(agent_to_predicts[idcs])
            ctx_interesting_agents.append(interesting_agents[idcs])
            ctx_shapes.append(shapes[idcs])
            ctx_agenttypes.append(agenttypes[idcs])   

        data = dict()
        data['city'] = city 
        data['trajs'] = ctx_trajs
        data['steps'] = ctx_steps 
        data['vels'] = ctx_vels
        data['psirads'] = ctx_psirads
        data['track_ids'] = ctx_track_ids 
        data['agent_to_predicts'] = ctx_agent_to_predicts 
        data['interesting_agents'] = ctx_interesting_agents
        data['shapes'] = ctx_shapes
        data['agenttypes'] = ctx_agenttypes

        return data

    def get_obj_feats(self, data, idx):
        scored_ctrs = []
        scored_indices = []
        for i in range(len(data['trajs'])):
            if 8 in data['steps'][i] and 9 in data['steps'][i]:
                present_index = list(data['steps'][i]).index(9)
                scored_ctrs.append(np.expand_dims(data['trajs'][i][present_index], axis=0))
                scored_indices.append(i)

        scored_ctrs = np.concatenate(scored_ctrs, axis=0)
        centroid = np.mean(scored_ctrs, axis=0)
        dist_to_centroid = np.linalg.norm(scored_ctrs - np.expand_dims(centroid, axis=0), ord=2, axis=-1)
        closest_centroid = np.argmin(dist_to_centroid, axis=0)
        orig_idx = scored_indices[closest_centroid]

        orig = data['trajs'][orig_idx][9].copy().astype(np.float32)
        pre = data['trajs'][orig_idx][8] - orig 
        theta = np.pi - np.arctan2(pre[1], pre[0])

        # rotation matrix for rotating scene
        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        feats, feat_locs, feat_vels, has_obss = [], [], [], []
        feat_psirads, feat_shapes, ctrs, feat_agenttypes, feat_agenttopredicts, feat_interestingagents, track_ids = [], [], [], [], [], [], []
        is_valid_agent = []

        for traj, step, vel, psirad, track_id, agent_to_predict, interesting_agent, shape, agenttype in zip(data['trajs'], data['steps'], data['vels'], data['psirads'], data['track_ids'], data['agent_to_predicts'], data['interesting_agents'], data['shapes'], data['agenttypes']):
            if 9 not in step:
                is_valid_agent.append(0)
                if 1 in agent_to_predict:
                    print("This should not happen!")
                    exit(0)
                continue

            is_valid_agent.append(1)

            has_obs = np.zeros(10, bool)

            # observation + future horizon
            idcs = step.argsort()
            step = step[idcs]
            traj = traj[idcs]
            vel = vel[idcs]
            agenttype = agenttype[idcs]
            psirad = psirad[idcs]
            track_id = track_id[idcs]
            agent_to_predict = agent_to_predict[idcs]
            interesting_agent = interesting_agent[idcs]
            shape = shape[idcs]
            has_obs[step] = 1

            # only observation horizon
            obs_step = step[step < 10]
            obs_idcs = obs_step.argsort()
            obs_step = obs_step[obs_idcs]

            # take contiguous past to be the past
            for i in range(len(obs_step)):
                if obs_step[i] == 10 - len(obs_step) + i:
                    break
            step = step[i:]
            traj = traj[i:]
            vel = vel[i:]
            agenttype = agenttype[i:]
            psirad = psirad[i:]
            shape = shape[i:]

            feat = np.zeros((10, 2), np.float32)
            feat_vel = np.zeros((10, 2), np.float32)
            feat_agenttype = np.zeros((10, 1), np.float32)
            feat_psirad = np.zeros((10, 1), np.float32)
            feat_trackid = np.zeros((10, 1), np.float32)
            feat_agenttopredict = np.zeros((10, 1), np.float32)
            feat_interestingagent = np.zeros((10, 1), np.float32)
            feat_shape = np.zeros((10, 2), np.float32)

            # center and rotate positions, rotate velocities
            feat[step] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            feat_vel[step] = np.matmul(rot, vel.T).T

            # recalculate yaw angles
            feat_agenttype[step] = agenttype
            feat_trackid[step] = track_id
            feat_agenttopredict[step] = agent_to_predict 
            feat_interestingagent[step] = interesting_agent
            feat_shape[step] = np.nan_to_num(shape) * feat_agenttype[step] # 0 out if not a car 

            if feat_agenttype[9] != 0:
                for j in range(len(psirad)):
                    psirad[j, 0] = psirad[j, 0] + theta
                    # angle now between -pi and 2pi
                    if psirad[j, 0] >= (2 * math.pi):
                        psirad[j, 0] = psirad[j] % (2 * math.pi)
                    # if between pi and 2pi
                    if psirad[j, 0] > math.pi:
                        psirad[j, 0] = -1 * ((2 * math.pi) - psirad[j, 0])
            # pedestrian/bicycle does not have yaw angle; use velocity to infer yaw when available; otherwise set to 0
            # velocity is already in se(2) transformed coordinates
            else:
                vel_transformed = feat_vel[step]
                assert len(psirad) == len(vel_transformed)
                for j in range(len(psirad)):
                    speed_j = math.sqrt(vel_transformed[j, 0] ** 2 + vel_transformed[j, 1] ** 2)
                    if speed_j == 0:
                        psirad[j, 0] = 0.
                    else:
                        psirad[j, 0] = round(sign_func(vel_transformed[j, 1]) * math.acos(vel_transformed[j, 0] / speed_j), 3)
            
            assert not np.any(np.isnan(psirad))
            feat_psirad[step] = psirad

            # ctrs contains the centers at the present timestep
            ctrs.append(feat[9, :].copy())

            feat_loc = np.copy(feat)
            # feat contains trajectory offsets
            feat[1:, :] -= feat[:-1, :]
            feat[step[0], :] = 0 

            feats.append(feat)
            feat_locs.append(feat_loc)
            feat_vels.append(feat_vel)
            feat_agenttypes.append(feat_agenttype)
            feat_psirads.append(feat_psirad)
            track_ids.append(feat_trackid)
            feat_agenttopredicts.append(feat_agenttopredict)
            feat_interestingagents.append(feat_interestingagent)
            feat_shapes.append(feat_shape)
            has_obss.append(has_obs)

        ctrs = np.asarray(ctrs, np.float32)
        feats = np.asarray(feats, np.float32)
        feat_locs = np.asarray(feat_locs, np.float32)
        feat_vels = np.asarray(feat_vels, np.float32)
        feat_agenttypes = np.asarray(feat_agenttypes, np.float32)
        feat_psirads = np.asarray(feat_psirads, np.float32)
        
        track_ids = np.asarray(track_ids, np.float32)
        feat_agenttopredicts= np.asarray(feat_agenttopredicts, np.float32)
        feat_interestingagents = np.asarray(feat_interestingagents, np.float32)
        
        feat_shapes = np.asarray(feat_shapes, np.float32)
        has_obss = np.asarray(has_obss, np.float32)
        is_valid_agent = np.asarray(is_valid_agent, bool)

        # Check that there are no nans
        assert theta <= (2 * math.pi)
        assert theta >= 0
        assert not np.any(np.isnan(ctrs))
        assert not np.any(np.isnan(feats))
        assert not np.any(np.isnan(feat_locs))
        assert not np.any(np.isnan(feat_vels))
        assert not np.any(np.isnan(feat_agenttypes))
        assert not np.any(np.isnan(feat_psirads))
        assert not np.any(np.isnan(track_ids))
        assert not np.any(np.isnan(feat_agenttopredicts))
        assert not np.any(np.isnan(feat_interestingagents))
        assert not np.any(np.isnan(feat_shapes))
        assert not np.any(np.isnan(has_obss))
        assert not np.any(np.isnan(is_valid_agent))

        data['feats'] = feats 
        data['ctrs'] = ctrs 
        data['feat_locs'] = feat_locs
        data['feat_vels'] = feat_vels 
        data['feat_agenttypes'] = feat_agenttypes
        data['feat_psirads'] = feat_psirads 
        data['track_ids'] = track_ids
        data['feat_agenttopredicts'] = feat_agenttopredicts
        data['feat_interestingagents'] = feat_interestingagents
        data['feat_shapes'] = feat_shapes
        data['has_obss'] = has_obss
        data['orig'] = orig 
        data['theta'] = theta 
        data['rot'] = rot 
        data['is_valid_agent'] = is_valid_agent

        return data


    def get_lane_graph(self, data):
        # Note that we process the full lane graph -- we do not have a prediction range like LaneGCN
        map_path = os.path.join(self.config['maps'], data['city'] + '.osm')
        map = lanelet2.io.load(map_path, self.projector)
        routing_graph = lanelet2.routing.RoutingGraph(map, self.traffic_rules)

        # build node features
        lane_ids, ctrs, feats = [], [], []
        centerlines, left_boundaries, right_boundaries = [], [], []
        for ll in map.laneletLayer:
            left_boundary = np.zeros((len(ll.leftBound), 2))
            right_boundary  = np.zeros((len(ll.rightBound), 2))

            for i in range(len(ll.leftBound)):
                left_boundary[i][0] = copy.deepcopy(ll.leftBound[i].x)
                left_boundary[i][1] = copy.deepcopy(ll.leftBound[i].y)

            for i in range(len(ll.rightBound)):
                right_boundary[i][0] = copy.deepcopy(ll.rightBound[i].x)
                right_boundary[i][1] = copy.deepcopy(ll.rightBound[i].y)
            
            # computes centerline with min(max(M,N), 10) data points per lanelet
            centerline, _ = compute_midpoint_line(left_boundary, right_boundary, min(10, max(left_boundary.shape[0], right_boundary.shape[0])))
            centerline = copy.deepcopy(centerline)            

            # process lane centerline in same way as agent trajectories
            centerline = np.matmul(data['rot'], (centerline - data['orig'].reshape(-1, 2)).T).T
            left_boundary = np.matmul(data['rot'], (left_boundary - data['orig'].reshape(-1, 2)).T).T
            right_boundary = np.matmul(data['rot'], (right_boundary - data['orig'].reshape(-1, 2)).T).T
            
            num_segs = len(centerline) - 1
            # locations between the centerline segments
            ctrs.append(np.asarray((centerline[:-1] + centerline[1:]) / 2.0, np.float32))
            # distances between the centerline segments
            feats.append(np.asarray(centerline[1:] - centerline[:-1], np.float32))
            lane_ids.append(ll.id)
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
        
        for i, lane_id in enumerate(lane_ids):
            lane = map.laneletLayer[lane_id]
            idcs = node_idcs[i]

            # points to the predecessor
            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]
            if len(routing_graph.previous(lane)) > 0:
                for prev_lane in routing_graph.previous(lane):
                    if prev_lane.id in lane_ids:
                        j = lane_ids.index(prev_lane.id)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])

            # points to the successor
            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]
            if len(routing_graph.following(lane)) > 0:
                for foll_lane in routing_graph.following(lane):
                    if foll_lane.id in lane_ids:
                        j = lane_ids.index(foll_lane.id)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])

        # we now compute lane-level features
        # lane indices
        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)

        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = map.laneletLayer[lane_id]

            # compute lane_id pairs of predecessor [u,v]
            if len(routing_graph.previous(lane)) > 0:
                for prev_lane in routing_graph.previous(lane):
                    if prev_lane.id in lane_ids:
                        j = lane_ids.index(prev_lane.id)
                        pre_pairs.append([i, j])

            # compute lane_id pairs of successor [u,v]
            if len(routing_graph.following(lane)) > 0:
                for foll_lane in routing_graph.following(lane):
                    if foll_lane.id in lane_ids:
                        j = lane_ids.index(foll_lane.id)
                        suc_pairs.append([i, j])

            # compute lane_id pairs of left [u,v]
            if routing_graph.left(lane) is not None:
                if routing_graph.left(lane).id in lane_ids:
                    j = lane_ids.index(routing_graph.left(lane).id)
                    left_pairs.append([i, j])

            # compute lane_id pairs of right [u,v]
            if routing_graph.right(lane) is not None:
                if routing_graph.right(lane).id in lane_ids:
                    j = lane_ids.index(routing_graph.right(lane).id)
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

    def __getitem__(self, idx):
        if self.config['preprocess']:
            data = np.load(os.path.join(self.preprocess_path, "{}.p".format(idx)), allow_pickle=True)

            new_data = dict()
            for key in ['idx', # unique integer assigned to each scene (assigned when building mapping dictionary)
                        'track_ids', # track id over the track (needed for submission to multi-agent challenge)
                        'feats', # position features (offsets) in SE(2)-transformed coordinate system (past)
                        'feat_locs', # location features (not offsets but absolute positions in SE(2)-transformed coordinate system) (past)
                        'feat_vels', # velocity features, in SE(2)-transformed coordinate system (past)
                        'feat_agenttypes', # agent_type (either car or pedestrian/bicycle, past)
                        'feat_psirads', # yaw angle features, in SE(2)-transformed coordinate system (past)
                        'feat_shapes', # vehicle [length,width] (past)
                        'feat_agenttopredicts', # whether the agent needs to be predicted for multiagent challenge
                        'feat_interestingagents', # whether the agent is an "interesting" agent, according to INTERACTION dataset
                        'has_obss', # past timestep exists mask (past)
                        'theta', # angle for rotating scene
                        'rot', # rotation matrix for rotating scene 
                        'orig', # origin of scene (taken to be centroid vehicle)
                        'ctrs', # agent centers at the present timestep in SE(2)-transformed coordinate system
                        'graph']: # lane graph

                if key in data:
                    new_data[key] = ref_copy(data[key])
            data = new_data

            return data 

        data = self.read_interaction_data(idx)
        data = self.get_obj_feats(data, idx)
        data['idx'] = idx
        data['graph'] = self.get_lane_graph(data)

        return data