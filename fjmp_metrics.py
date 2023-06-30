import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist
import torch
import pickle
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

### FROM INTERACTION DATASET REPO
"""
 This function returns the list of origins of circles for the given vehicle at all predicted timestamps and modalities.
 x, y, and yaw has the same shape (T, Modality).
 l, w are scalars represents the length and width of the vehicle.
 The output has the shape (T, Modality, c, 2) where c is the number of circles and c is determined by the length of the given vehicle.
"""
def return_circle_list(x, y, l, w, yaw):
    r = w/np.sqrt(2)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    if l < 4.0:
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c = [c1, c2]
    elif l >= 4.0 and l < 8.0:
        c0 = [x, y]
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c = [c0, c1, c2]
    else:
        c0 = [x, y]
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c3 = [x-(l-w)/2*cos_yaw/2, y-(l-w)/2*sin_yaw/2]
        c4 = [x+(l-w)/2*cos_yaw/2, y+(l-w)/2*sin_yaw/2]
        c = [c0, c1, c2, c3, c4]
    for i in range(len(c)):
        c[i] = np.stack(c[i], axis=-1)
    c = np.stack(c, axis=-2)
    return c

### FROM INTERACTION DATASET REPO
"""
 This function returns the threshold for collision. If any of two circles' origins' distance between two vehicles is lower than the threshold, it is considered as they have a collision at that timestamp.
 w1, w2 is scalar value which represents the width of vehicle 1 and vehicle 2.
"""
def return_collision_threshold(w1, w2):
    return (w1 + w2) / np.sqrt(3.8)

### Calculates the estimated heading of the future trajectories
### Since heading (i.e., yaw) is not explicitly predicted, we use heuristics to smooth the heading estimates
def get_psirad_pred(loc_pred, heading_present, ctrs):
    N = loc_pred.shape[0]
    T = loc_pred.shape[1]
    num_joint_modes = loc_pred.shape[2]
    DIST_THRESHOLD_1 = 5
    DIST_THRESHOLD_2 = 2
    
    ctrs = np.stack([ctrs]*num_joint_modes, axis=1)

    # more coarse velocity estimate to prevent sudden changes in heading
    vel_pred = np.zeros_like(loc_pred)
    for i in range(0, T, 6):
        low = i
        high = i+6
        if low == 0:
            vel_pred[:, low:high] = np.expand_dims(loc_pred[:,high-1]-ctrs, axis=1) / (0.1 * (high - low))
        else:
            vel_pred[:, low:high] = np.expand_dims(loc_pred[:,high-1]-loc_pred[:,low-1], axis=1) / (0.1 * (high - low))

    # if total distance travelled by vehicle is less than 5 metres, use same velocity vector
    dist_travelled = np.linalg.norm(loc_pred[:, 1:] - loc_pred[:, :-1], axis=-1, ord=2)
    dist_travelled_0 = np.linalg.norm(loc_pred[:, 0] - ctrs, axis=-1, ord=2)
    dist_travelled = np.sum(np.concatenate([np.expand_dims(dist_travelled_0, axis=1), dist_travelled], axis=1), axis=1)
    # Shape: [N, num_joint_modes]
    overwrite_pred_mask_1 = dist_travelled <= DIST_THRESHOLD_1

    if overwrite_pred_mask_1.sum() != 0:
        for i in range(N):
            if overwrite_pred_mask_1[i].sum() == 0:
                continue 
            for mode in range(num_joint_modes):
                if overwrite_pred_mask_1[i, mode].sum() == 0:
                    continue
                else:
                    vel_pred[i, :, mode, :] = (loc_pred[i, -1, mode, :] - ctrs[i, mode, :]) / (0.1 * loc_pred.shape[1])

    # if total distance travelled by vehicles is less than 2 metres, overwrite predicted heading to the present heading.
    overwrite_pred_mask_2 = dist_travelled <= DIST_THRESHOLD_2
    
    psirad_pred = np.sign(vel_pred[:,:,:,1]) * np.arccos(vel_pred[:,:,:,0] / (np.linalg.norm(vel_pred, ord=2, axis=-1)))
    
    if overwrite_pred_mask_2.sum() != 0:
        for i in range(N):
            if overwrite_pred_mask_2[i].sum() == 0:
                continue 
            for mode in range(num_joint_modes):
                if overwrite_pred_mask_2[i, mode].sum() == 0:
                    continue
                else:
                    # ground-truth heading from t=observation_steps-1
                    psirad_pred[i, :, mode] = heading_present[i]    

    return psirad_pred

# Note: Scene collision rate (SCR) is defined as the proportion of modalities where at least one pairwise collision exists.
# Note: SCR is only computed for INTERACTION, as bounding box information is required to compute collision existence, which is not provided in AV2
def compute_scene_collision_rate(n_scenarios, loc_pred, batch_idxs, theta_all, feat_psirads, shapes, ctrs, gt_locs):
    
    # Each value appended into this list will be the SCR of the given scene (num_modalities_with_a_collision / num_modalities)
    scrs = []
    for scene in tqdm(range(n_scenarios)):
        loc_pred_scene = loc_pred[batch_idxs == scene]
        N = loc_pred_scene.shape[0]

        # If only one agent in the scene, no collisions in scene by definition
        if N <= 1:
            scrs.append(0)
            continue
        
        # [N, 40, 1]
        # convert the present heading (in se(2)-transformed coordinates) back to ground-truth coordinate system
        heading_present = feat_psirads[batch_idxs == scene].reshape((-1, 40))[:, 9]
        theta_all_scene = theta_all[batch_idxs == scene]

        # original is between [-pi, pi], now between -3pi and pi
        for i in range(len(heading_present)):
            heading_present[i] = heading_present[i] - theta_all_scene[i]
            
            # angle now between -pi and 2pi
            if heading_present[i] < -1 * (math.pi):
                heading_present[i] = heading_present[i] % (2 * math.pi)
            # if between pi and 2pi
            if heading_present[i] > math.pi:
                heading_present[i] = -1 * ((2 * math.pi) - heading_present[i])

        shape_scene = shapes[batch_idxs == scene]
        ctrs_scene = ctrs[batch_idxs == scene]
        gt_locs_scene = gt_locs[batch_idxs == scene]
        psirad_pred_scene = get_psirad_pred(loc_pred_scene, heading_present, ctrs_scene)
        
        bad_ks = set()
        for i in range(1, N):
            for j in range(i):
                # [T, Modality, c, 2]
                circle_list_i = return_circle_list(loc_pred_scene[i, :, :, 0], loc_pred_scene[i, :, :, 1], shape_scene[i, 0], shape_scene[i, 1], psirad_pred_scene[i])
                circle_list_j = return_circle_list(loc_pred_scene[j, :, :, 0], loc_pred_scene[j, :, :, 1], shape_scene[j, 0], shape_scene[j, 1], psirad_pred_scene[j])

                thresh = return_collision_threshold(shape_scene[i, 1], shape_scene[j, 1])

                for k in range(6):
                    dist = np.expand_dims(circle_list_i[:, k, :, :], axis=2) - np.expand_dims(circle_list_j[:, k, :, :], axis=1)
                    # compute pairwise distances
                    dist = np.linalg.norm(dist, axis=-1, ord=2)
                    
                    bad = dist < thresh
                    if bad.sum() >= 1:
                        bad_ks.add(k)

        scrs.append(len(bad_ks) / 6)

    return sum(scrs) / len(scrs)

### Longitudinal threshold calculation for Miss Rate computation on INTERACTION benchmark
# Note that the formula is slightly different to that proposed in https://github.com/interaction-dataset/INTERPRET_challenge_multi-agent
# See https://github.com/interaction-dataset/INTERPRET_challenge_single-agent/issues/1 for details
def compute_longitudinal_threshold(vels):
    mag = np.linalg.norm(vels, axis=-1, ord=2)
    
    thresh = np.zeros(vels.shape[0])
    thresh[mag < 1.4] = 1
    thresh[mag > 11] = 2
    thresh[thresh == 0] = 1 + (mag[thresh == 0] - 1.4) / (11 - 1.4)

    return thresh

# This function computes the Scene Miss Rate (SMR) according to the Argoverse 2 definition of a "miss"
def compute_av2_scene_miss_rate(n_scenarios, loc_pred, batch_idxs, gt_psirads, gt_locs, gt_vels):
    
    smrs = []
    for scene in range(n_scenarios):
        loc_pred_scene = loc_pred[batch_idxs == scene]
        N = loc_pred_scene.shape[0]

        assert N >= 1
        
        gt_locs_scene = gt_locs[batch_idxs == scene]
        last_gt_locs_scene = gt_locs_scene[:, -1].reshape(-1, 6, 1, 2)
        last_loc_pred_scene = loc_pred_scene[:, -1].reshape(-1, 6, 1, 2)

        diff = np.linalg.norm(last_gt_locs_scene - last_loc_pred_scene, axis=-1)[:,:,0]
        
        bad = diff > 2   
        bad_tot = np.sum(bad, axis=0) / bad.shape[0]
        
        smrs.append(bad_tot.min())

    return sum(smrs) / len(smrs)

# This function computes the Scene Miss Rate (SMR) according to the INTERACTION benchmark metric
# The min scene miss rate of a scene is: min over modes (proportion of agent predictions that "miss" ground-truth)
# The min scene miss rate is then averaged over the scenes.
# Note that the trajectories passed into this function all have a ground-truth position at the last timestep.
def compute_scene_miss_rate(n_scenarios, loc_pred, batch_idxs, gt_psirads, gt_locs, gt_vels):
    
    # Each value appended to this list in the min scene miss rate for a particular scene
    smrs = []
    for scene in range(n_scenarios):
        loc_pred_scene = loc_pred[batch_idxs == scene]
        N = loc_pred_scene.shape[0]

        assert N >= 1
        
        gt_psirads_scene = gt_psirads[batch_idxs == scene]
        gt_locs_scene = gt_locs[batch_idxs == scene]
        gt_vels_scene = gt_vels[batch_idxs == scene]

        last_gt_locs_scene = gt_locs_scene[:, -1].reshape(-1, 6, 1, 2)
        last_loc_pred_scene = loc_pred_scene[:, -1].reshape(-1, 6, 1, 2)
        last_gt_psirads_scene = gt_psirads_scene[:, -1, 0]
        last_gt_vels_scene = gt_vels_scene[:, -1].reshape(-1, 1, 1, 2)

        # build rotation matrix for aligning predictions
        rot = np.zeros((N, 2, 2))
        for i in range(N):
            rot[i] = np.asarray([
            [np.cos(last_gt_psirads_scene[i]), -np.sin(last_gt_psirads_scene[i])],
            [np.sin(last_gt_psirads_scene[i]), np.cos(last_gt_psirads_scene[i])]], np.float32)
        rot = rot.reshape(-1, 1, 2, 2)
        
        # [N, 6, 1, 2] * [N, 1, 2, 2]
        rot_last_gt_locs_scene = np.matmul(last_gt_locs_scene, rot)
        rot_last_loc_pred_scene = np.matmul(last_loc_pred_scene, rot)

        diff = np.abs(rot_last_gt_locs_scene - rot_last_loc_pred_scene)[:,:,0,:]

        # start with lateral miss rate
        bad_y = diff[:, :, 1] > 1

        # now longitudinal miss rate
        thresh = compute_longitudinal_threshold(last_gt_vels_scene.reshape(-1, 2))
        thresh = np.stack([thresh]*6, axis=1)
        bad_x = diff[:, :, 0] > thresh    

        bad = np.logical_or(bad_x, bad_y)
        # [6,]
        bad_tot = np.sum(bad, axis=0) / bad.shape[0]
        
        smrs.append(bad_tot.min())

    return sum(smrs) / len(smrs)

'''
Calculate joint forecasting metrics of all agents who have ground-truth waypoint at last position
'''
def calc_metrics(results, config, mask, identifier):
    
    n_scenes_before_mask = np.unique(results['batch_idxs']).shape[0]

    # first apply the mask to the necessary data
    gt_locs_all = results['gt_locs_all'][mask]
    has_preds_all = results['has_preds_all'][mask].astype(bool)
    batch_idxs = results['batch_idxs'][mask]
    gt_psirads_all = results['gt_psirads_all'][mask]
    feat_psirads_all = results['feat_psirads_all'][mask]
    gt_vels_all = results['gt_vels_all'][mask]
    theta_all = results['theta_all'][mask]
    ctrs_all = results['gt_ctrs_all'][mask]
    if config['dataset'] == "interaction":
        shapes_all = results['shapes_all'][mask]
    if config['proposal_header']:
        proposals = results["proposals_all"][mask]
    if (not config['two_stage_training']) or (config['two_stage_training'] and config['training_stage'] == 2):
        loc_pred = results['loc_pred'][mask]  
    
    n_scenarios = np.unique(batch_idxs).shape[0]
    scenarios = np.unique(batch_idxs).astype(int)

    if identifier == 'int':
        # these are the best modes from the regular evaluation, per-scene
        best_modes = np.load("best_modes_{}.npy".format(config["config_name"]))[scenarios].astype(int)
    
    # sanity check
    last = torch.from_numpy(has_preds_all).float() + 0.1 * torch.arange(config["prediction_steps"]).float().reshape(1, -1) / config["prediction_steps"]
    max_last, last_idcs = last.max(1)
    # last index for each trajectory in the batch
    last_idcs = last_idcs.numpy()
    assert np.all(last_idcs == (config["prediction_steps"] - 1))
    
    has_preds_all_mask = np.reshape(has_preds_all, has_preds_all.shape + (1,))
    # [N, 30, 6]
    has_preds_all_mask = np.broadcast_to(has_preds_all_mask, has_preds_all_mask.shape[:2] + (config["num_joint_modes"],))  
      
    if config["proposal_header"] and identifier == 'reg':
        num_proposals = proposals.shape[2]
        # [N, 30, 6, 2]
        gt_locs_all_proposals = np.stack([gt_locs_all]*num_proposals, axis=2)
        has_preds_all_mask_proposals = np.reshape(has_preds_all, has_preds_all.shape + (1,))
        has_preds_all_mask_proposals = np.broadcast_to(has_preds_all_mask_proposals, has_preds_all_mask_proposals.shape[:2] + (num_proposals,)) 
        
        # calculate FDE, ADE of proposals 
        # [N, 30, 6, 2]
        mse_error = (proposals - gt_locs_all_proposals)**2
        # [N, 30, 6]
        euclidean_rmse = np.sqrt(mse_error.sum(-1))    
        
        # zero out timesteps without ground-truth data point
        euclidean_rmse_filtered = np.zeros(euclidean_rmse.shape)
        euclidean_rmse_filtered[has_preds_all_mask_proposals] = euclidean_rmse[has_preds_all_mask_proposals]
    
        # mean over the agents then min over the num_proposals samples then mean over the scenarios
        mean_FDE = np.zeros((n_scenarios, num_proposals))
        mean_ADE = np.zeros((n_scenarios, num_proposals)) 

        for j, i in enumerate(scenarios):
            i = int(i)
            has_preds_all_i = has_preds_all[batch_idxs == i]
            euclidean_rmse_filtered_i = euclidean_rmse_filtered[batch_idxs == i]
            mean_FDE[j] = euclidean_rmse_filtered_i[:, -1].mean(0)
            mean_ADE[j] = euclidean_rmse_filtered_i.sum((0, 1)) / has_preds_all_i.sum()

        pFDE = mean_FDE.min(1).mean()
        pADE = mean_ADE.min(1).mean()

    
    if (not config["two_stage_training"]) or (config["two_stage_training"] and config["training_stage"] == 2):
        # [N, 30, 6, 2]
        num_joint_modes = loc_pred.shape[2]
        gt_locs_all = np.stack([gt_locs_all]*num_joint_modes, axis=2)

        if config['dataset'] == 'interaction' and config['mode'] == 'eval' and identifier == 'reg':
            scr = compute_scene_collision_rate(n_scenarios, loc_pred, batch_idxs, theta_all, feat_psirads_all, shapes_all, ctrs_all, gt_locs_all)
        else:
            scr=0
        if config['mode'] == 'eval' and identifier == 'reg':
            smr = compute_scene_miss_rate(n_scenarios, loc_pred, batch_idxs, gt_psirads_all, gt_locs_all, gt_vels_all)
            smr_av2 = compute_av2_scene_miss_rate(n_scenarios, loc_pred, batch_idxs, gt_psirads_all, gt_locs_all, gt_vels_all)
        else:
            smr = 0
            smr_av2 = 0
            
        # [N, 30, 6, 2]
        mse_error = (loc_pred - gt_locs_all)**2
        # [N, 30, 6]
        euclidean_rmse = np.sqrt(mse_error.sum(-1))   
        
        euclidean_rmse_filtered = np.zeros(euclidean_rmse.shape)
        euclidean_rmse_filtered[has_preds_all_mask] = euclidean_rmse[has_preds_all_mask]
    
        # mean over the agents then min over the num_joint_modes samples then mean over the scenarios
        mean_FDE = np.zeros((n_scenarios, num_joint_modes))
        mean_ADE = np.zeros((n_scenarios, num_joint_modes))
        
        for j, i in enumerate(scenarios):
            i = int(i)
            has_preds_all_i = has_preds_all[batch_idxs == i]
            euclidean_rmse_filtered_i = euclidean_rmse_filtered[batch_idxs == i]
            mean_FDE[j] = euclidean_rmse_filtered_i[:, -1].mean(0)
            mean_ADE[j] = euclidean_rmse_filtered_i.sum((0, 1)) / has_preds_all_i.sum()
        
        if identifier == 'reg':
            FDE = mean_FDE.min(1).mean()
            ADE = mean_ADE.min(1).mean()

            # initialize to -1
            best_modes = np.ones(n_scenes_before_mask) * -1
            best_modes[scenarios] = np.argmin(mean_FDE, axis=1)
            # We need to save the best modes, so that we evaluate the interactive metrics on the best modes.
            np.save("best_modes_{}.npy".format(config["config_name"]), best_modes)
        else:
            best_mode_FDE = np.take_along_axis(mean_FDE, best_modes.reshape((-1, 1)), 1)[:, 0]
            best_mode_ADE = np.take_along_axis(mean_ADE, best_modes.reshape((-1, 1)), 1)[:, 0]
            
            FDE = best_mode_FDE.mean()
            ADE = best_mode_ADE.mean()
    
    else:
        # default values for when training stage = 1
        FDE = 0
        ADE = 0
        scr = 0
        smr = 0
        smr_av2 = 0

    if config["learned_relation_header"]:
        # We now compute accuracy of relation header
        ig_preds = np.argmax(results['ig_preds'], axis=1)
        relation_accuracy = np.mean(ig_preds == results['ig_labels_all'])

        proportion_no_edge = np.sum(ig_preds == 0) / ig_preds.shape[0]
        edge_mask_0 = results['ig_labels_all'] == 0
        edge_mask_1 = results['ig_labels_all'] == 1
        edge_mask_2 = results['ig_labels_all'] == 2

        edge_accuracy_0 = np.mean(ig_preds[edge_mask_0] == 0)
        edge_accuracy_1 = np.mean(ig_preds[edge_mask_1] == 1)
        edge_accuracy_2 = np.mean(ig_preds[edge_mask_2] == 2)            

    results = {
        "FDE": FDE,
        "ADE": ADE}

    if config["proposal_header"] and identifier == 'reg':
        results["pFDE"] = pFDE
        results["pADE"] = pADE
    else:
        results["pFDE"] = 0
        results["pADE"] = 0
    
    results["n_scenarios"] = n_scenarios
    results["SCR"] = scr 
    results["SMR"] = smr
    results["SMR_AV2"] = smr_av2

    if config["learned_relation_header"]:
        results["Edge Accuracy"] = relation_accuracy
        results["Proportion No Edge"] = proportion_no_edge
        results["Edge Accuracy 0"] = edge_accuracy_0
        results["Edge Accuracy 1"] = edge_accuracy_1
        results["Edge Accuracy 2"] = edge_accuracy_2
    
    return results