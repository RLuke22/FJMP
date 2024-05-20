import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader
import dgl
import matplotlib.pyplot as plt
import re, csv

import pickle
from tqdm import tqdm
import argparse
import os, sys, time
import random
from pathlib import Path

from fjmp_dataloader_interaction import InteractionTestDataset
from fjmp_modules import *
from fjmp_utils import *
from dag_utils import *
from fjmp_metrics import *

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=['train', 'eval', 'eval_constant_velocity'], help='running mode : (train, eval, eval_constant_velocity)', default="train")
parser.add_argument("--dataset", choices=['interaction', 'argoverse2'], help='dataset : (interaction, argoverse2)', default="interaction")
parser.add_argument("--config_name", default="dev", help="a name to indicate the log path and model save path")
parser.add_argument("--num_edge_types", default=3, type=int, help='3 types: no-interaction, a-influences-b, b-influences-a')
parser.add_argument("--h_dim", default=128, type=int, help='dimension for the hidden layers of MLPs. Note that the GRU always has h_dim=256')
parser.add_argument("--num_joint_modes", default=6, type=int, help='number of scene-level modes')
parser.add_argument("--num_proposals", default=15, type=int, help='number of proposal modes')
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--max_epochs", default=50, type=int, help='maximum number of epochs')
parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
parser.add_argument("--decoder", choices=['dagnn', 'lanegcn'], help='decoder architecture : (dagnn, lanegcn)', default="dagnn")
parser.add_argument("--num_heads", default=1, type=int, help='number of heads in multi-head attention for decoder attention.')
parser.add_argument("--learned_relation_header", action="store_true", help="if true, network learns+predicts interaction graph with interaction graph predictor. Otherwise, ground-truth pseudolabels are used.")
parser.add_argument("--gpu_start", default=0, type=int, help='gpu device i, where training will occupy gpu device i,i+1,...,i+n_gpus-1')
parser.add_argument("--n_mapnet_layers", default=2, type=int, help='number of MapNet blocks')
parser.add_argument("--n_l2a_layers", default=2, type=int, help='number of L2A attention blocks')
parser.add_argument("--n_a2a_layers", default=2, type=int, help='number of A2A attention blocks')
parser.add_argument("--resume_training", action="store_true", help="continue training from checkpoint")
parser.add_argument("--proposal_coef", default=1, type=float, help="coefficient for proposal losses")
parser.add_argument("--rel_coef", default=100, type=float, help="coefficient for interaction graph prediction losses.")
parser.add_argument("--proposal_header", action="store_true", help="add proposal multitask training objective?")
parser.add_argument("--two_stage_training", action="store_true", help="train relation predictor first?")
parser.add_argument("--training_stage", default=1, type=int, help='1 or 2. Which training stage in 2 stage training?')
parser.add_argument("--ig", choices=['sparse', 'dense', 'm2i'], help='which interaction graph pseudolabels to use', default="sparse")
parser.add_argument("--focal_loss", action="store_true", help="use multiclass focal loss for relation header?")
parser.add_argument("--gamma", default=5, type=float, help="gamma parameter for focal loss.")
parser.add_argument("--weight_0", default=1., type=float, help="weight of class 0 for relation header.")
parser.add_argument("--weight_1", default=2., type=float, help="weight of class 1 for relation header.")
parser.add_argument("--weight_2", default=4., type=float, help="weight of class 2 for relation header.")
parser.add_argument("--teacher_forcing", action="store_true", help="use teacher forcing of influencer future predictions?")
parser.add_argument("--scheduled_sampling", action="store_true", help="use linear schedule curriculum for teacher forcing of influencer future predictions?")
parser.add_argument("--eval_training", action="store_true", help="run evaluation on training set?")
parser.add_argument("--supervise_vehicles", action="store_true", help="supervise only vehicles in loss function (for INTERACTION)?")
parser.add_argument("--train_all", action="store_true", help="train on both the train and validation sets?")
parser.add_argument("--no_agenttype_encoder", action="store_true", help="encode agent type in FJMP encoder? Only done for Argoverse 2 as INTERACTION only predicts vehicle trajectories.")

args = parser.parse_args()

GPU_START = args.gpu_start

import horovod.torch as hvd 
from torch.utils.data.distributed import DistributedSampler
from mpi4py import MPI

comm = MPI.COMM_WORLD
hvd.init()
os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank() + GPU_START)
dev = 'cuda:{}'.format(0)
torch.cuda.set_device(0)

seed = hvd.rank()
set_seeds(seed)

class FJMP(torch.nn.Module):
    def __init__(self, config):
        super(FJMP, self).__init__()
        self.config = config
        self.dataset = config["dataset"]
        self.num_train_samples = config["num_train_samples"]
        self.num_val_samples = config["num_val_samples"]
        self.num_test_samples = config["num_test_samples"]
        self.num_agenttypes = config["num_agenttypes"]
        self.switch_lr_1 = config["switch_lr_1"]
        self.switch_lr_2 = config["switch_lr_2"]
        self.lr_step = config["lr_step"]
        self.mode = config["mode"]
        self.input_size = config["input_size"]
        self.observation_steps = config["observation_steps"]
        self.prediction_steps = config["prediction_steps"]
        self.num_edge_types = config["num_edge_types"]
        self.h_dim = config["h_dim"]
        self.num_joint_modes = config["num_joint_modes"]
        self.num_proposals = config["num_proposals"]
        self.learning_rate = config["lr"]
        self.max_epochs = config["max_epochs"]
        self.log_path = config["log_path"]
        self.batch_size = config["batch_size"]
        self.decoder = config["decoder"]
        self.num_heads = config["num_heads"]
        self.learned_relation_header = config["learned_relation_header"]
        self.resume_training = config["resume_training"]
        self.proposal_coef = config["proposal_coef"]
        self.rel_coef = config["rel_coef"]
        self.proposal_header = config["proposal_header"]
        self.two_stage_training = config["two_stage_training"]
        self.training_stage = config["training_stage"]
        self.ig = config["ig"]
        self.focal_loss = config["focal_loss"]
        self.gamma = config["gamma"]
        self.weight_0 = config["weight_0"]
        self.weight_1 = config["weight_1"]
        self.weight_2 = config["weight_2"]
        self.teacher_forcing = config["teacher_forcing"]
        self.scheduled_sampling = config["scheduled_sampling"]
        self.eval_training = config["eval_training"]
        self.supervise_vehicles = config["supervise_vehicles"]
        self.no_agenttype_encoder = config["no_agenttype_encoder"]
        self.train_all = config["train_all"]
        
        if self.two_stage_training and self.training_stage == 2:
            self.pretrained_relation_header = None
        
        self.build()

    def build(self):
        self.feature_encoder = FJMPFeatureEncoder(self.config).to(dev)
        if self.learned_relation_header:
            self.relation_header = FJMPRelationHeader(self.config).to(dev)
        
        if self.proposal_header:
            self.proposal_decoder = FJMPTrajectoryProposalDecoder(self.config).to(dev)
        
        if (self.two_stage_training and self.training_stage == 2) or not self.two_stage_training:
            if self.decoder == 'dagnn':
                self.trajectory_decoder = FJMPAttentionTrajectoryDecoder(self.config).to(dev)
            elif self.decoder == 'lanegcn':
                self.trajectory_decoder = LaneGCNHeader(self.config).to(dev)

    def process(self, data):
        num_actors = [len(x) for x in data['feats']]
        num_edges = [int(n * (n-1) / 2) for n in num_actors]

        # LaneGCN processing 
        # ctrs gets copied once for each agent in scene, whereas actor_ctrs only contains one per scene
        # same data, but different format so that it is compatible with LaneGCN L2A/A2A function     
        actor_ctrs = gpu(data["ctrs"])
        lane_graph = graph_gather(to_long(gpu(data["graph"])), self.config)
        # unique index assigned to each scene
        scene_idxs = torch.Tensor([idx for idx in data['idx']])

        graph = data["graph"]

        world_locs = [x for x in data['feat_locs']]
        world_locs = torch.cat(world_locs, 0)

        has_obs = [x for x in data['has_obss']]
        has_obs = torch.cat(has_obs, 0)

        locs = [x for x in data['feats']]
        locs = torch.cat(locs, 0)

        vels = [x for x in data['feat_vels']]
        vels = torch.cat(vels, 0)

        psirads = [x for x in data['feat_psirads']]
        psirads = torch.cat(psirads, 0)

        track_ids  = [x for x in data['track_ids']]
        track_ids = torch.cat(track_ids, 0)[:, self.observation_steps - 1, 0]

        agenttopredicts  = [x for x in data['feat_agenttopredicts']]
        agenttopredicts = torch.cat(agenttopredicts, 0)[:, self.observation_steps - 1, 0]

        interestingagents = [x for x in data['feat_interestingagents']]
        interestingagents = torch.cat(interestingagents, 0)[:, self.observation_steps - 1, 0]

        agenttypes = [x for x in data['feat_agenttypes']]
        agenttypes = torch.cat(agenttypes, 0)[:, self.observation_steps - 1, 0]
        agenttypes = torch.nn.functional.one_hot(agenttypes.long(), self.num_agenttypes)

        shapes = [x for x in data['feat_shapes']]
        shapes = torch.cat(shapes, 0)

        feats = torch.cat([locs, vels, psirads], dim=2)

        ctrs = [x for x in data['ctrs']]
        ctrs = torch.cat(ctrs, 0)

        orig = [x.view(1, 2) for j, x in enumerate(data['orig']) for i in range(num_actors[j])]
        orig = torch.cat(orig, 0)

        rot = [x.view(1, 2, 2) for j, x in enumerate(data['rot']) for i in range(num_actors[j])]
        rot = torch.cat(rot, 0)

        theta = torch.Tensor([x for j, x in enumerate(data['theta']) for i in range(num_actors[j])])
        
        batch_idxs = []
        batch_idxs_edges = []
        actor_idcs = []
        sceneidx_to_batchidx_mapping = {}
        count_batchidx = 0
        count = 0
        for i in range(len(num_actors)):            
            batch_idxs.append(torch.ones(num_actors[i]) * count_batchidx)
            batch_idxs_edges.append(torch.ones(num_edges[i]) * count_batchidx)
            sceneidx_to_batchidx_mapping[int(scene_idxs[i].item())] = count_batchidx
            idcs = torch.arange(count, count + num_actors[i]).to(locs.device)
            actor_idcs.append(idcs)
            
            count_batchidx += 1
            count += num_actors[i]
        
        batch_idxs = torch.cat(batch_idxs).to(locs.device)
        batch_idxs_edges = torch.cat(batch_idxs_edges).to(locs.device)
        batch_size = torch.unique(batch_idxs).shape[0]

        dd = {
            'batch_size': batch_size,
            'batch_idxs': batch_idxs,
            'batch_idxs_edges': batch_idxs_edges, 
            'actor_idcs': actor_idcs,
            'actor_ctrs': actor_ctrs,
            'lane_graph': lane_graph,
            'feats': feats,
            'feat_psirads': psirads,
            'track_ids': track_ids,
            'agenttopredicts': agenttopredicts,
            'interestingagents': interestingagents,
            'ctrs': ctrs,
            'orig': orig,
            'rot': rot,
            'theta': theta,
            'scene_idxs': scene_idxs,
            'sceneidx_to_batchidx_mapping': sceneidx_to_batchidx_mapping,
            'agenttypes': agenttypes,
            'world_locs': world_locs,
            'has_obs': has_obs,
            'graph': graph,
            'shapes': shapes
        }

        # dd = data-dictionary
        return dd

    def _eval(self, test_loader, epoch):
        hvd.broadcast_parameters(self.state_dict(), root_rank=0)

        self.eval()
        # test results
        results = {}
        loc_preds, batch_idxs_all, scene_idxs_all = [], [], []
        psirad_preds_all, feat_psirads_all = [], []
        shapes_all, agenttypes_all, gt_ctrs_all = [], [], []
        theta_all = []
        track_ids_all = []
        agenttopredicts_all = []
        interestingagents_all = []            

        tot = 0
        with torch.no_grad():
            tot_log = self.num_test_samples // (self.batch_size * hvd.size())            
            for i, data in enumerate(test_loader):
                dd = self.process(data)
                
                dgl_graph = self.init_dgl_graph(dd['batch_idxs'], dd['ctrs'], dd['orig'], dd['rot'], dd['agenttypes'], dd['world_locs']).to(dev)
                dgl_graph = self.feature_encoder(dgl_graph, dd['feats'], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])

                if self.two_stage_training and self.training_stage == 2:
                    stage_1_graph = self.build_stage_1_graph(dgl_graph, dd['feats'], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])
                else:
                    stage_1_graph = None
                
                res = self.forward(dd["scene_idxs"], dgl_graph, stage_1_graph, dd['batch_idxs'], dd["actor_ctrs"], prop_ground_truth=0.)

                if i % 50 == 0:
                    print_("Test data: ", "{:.2f}%".format(i * 100 / tot_log))
                
                if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
                    loc_preds.append(res["loc_pred"].detach().cpu())                                           

                batch_idxs_all.append(dd['batch_idxs'].detach().cpu() + tot)
                scene_idxs_all.append(dd['scene_idxs'].detach().cpu())
                feat_psirads_all.append(dd['feat_psirads'].detach().cpu())
                track_ids_all.append(dd['track_ids'].detach().cpu())
                agenttopredicts_all.append(dd['agenttopredicts'].detach().cpu())
                interestingagents_all.append(dd['interestingagents'].detach().cpu())
                theta_all.append(dd['theta'].detach().cpu())
                shapes_all.append(dd['shapes'][:,0,:].detach().cpu())
                agenttypes_all.append(dd['agenttypes'].detach().cpu())
                gt_ctrs_all.append((torch.matmul(dd['ctrs'].unsqueeze(1), dd["rot"]).squeeze(1) + dd['orig']).detach().cpu())
                tot += dd['batch_size']
        
        print_('Calculating test set predictions...')
        
        results['batch_idxs'] = np.concatenate(batch_idxs_all)      
        results['scene_idxs'] = np.concatenate(scene_idxs_all)       
        results['feat_psirads_all'] = np.concatenate(feat_psirads_all, axis=0)
        results['track_ids_all'] = np.concatenate(track_ids_all, axis=0)
        results['agenttopredicts_all'] = np.concatenate(agenttopredicts_all, axis=0)
        results['interestingagents_all'] = np.concatenate(interestingagents_all, axis=0)
        results['theta_all'] = np.concatenate(theta_all, axis=0)
        results['shapes_all'] = np.concatenate(shapes_all, axis=0)
        results['agenttypes_all'] = np.concatenate(agenttypes_all, axis=0) 
        results['theta_all'] = np.concatenate(theta_all, axis=0)
        results['gt_ctrs_all'] = np.concatenate(gt_ctrs_all, axis=0)
        
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            results['loc_pred'] = np.concatenate(loc_preds, axis=0)    
        
        self.write_to_csv(results)

    def write_to_csv(self, results):

        with open(os.path.join('dataset_INTERACTION/mapping_test.pkl'), "rb") as f:
            mapping = pickle.load(f)
        filename_pattern = re.compile(r'^(\w+)_obs_(\d+).csv$')

        test_set_submission_dir = "sub"
        if not os.path.isdir(test_set_submission_dir):
            os.makedirs(test_set_submission_dir)  

        batch_idxs = results['batch_idxs']
        track_ids = results['track_ids_all']
        agenttopredicts = results['agenttopredicts_all']
        interestingagents = results['interestingagents_all']
        thetas = results['theta_all']
        agenttypes = results['agenttypes_all']
        scene_idxs = results['scene_idxs']
        ctrs = results['gt_ctrs_all']

        loc_preds = results['loc_pred']
        feat_psirads = results['feat_psirads_all']

        header = ('case_id', 'track_id', 'frame_id', 'timestamp_ms', 'agent_type', 'track_to_predict', 'interesting_agent', 'x1', 'y1', 'psi_rad1', 'x2', 'y2', 'psi_rad2', 'x3', 'y3', 'psi_rad3', 'x4', 'y4', 'psi_rad4', 'x5', 'y5', 'psi_rad5', 'x6', 'y6', 'psi_rad6')
        
        n_scenarios = np.unique(batch_idxs).shape[0]
        for i in tqdm(range(n_scenarios)):
            scene_idx = scene_idxs[i]
            csv_file = mapping[scene_idx]
            city = filename_pattern.match(csv_file).group(1)
            case_id = filename_pattern.match(csv_file).group(2)

            agenttopredicts_i = agenttopredicts[batch_idxs == i]
            interestingagents_i = interestingagents[batch_idxs == i][agenttopredicts_i == 1]
            thetas_i = thetas[batch_idxs ==i][agenttopredicts_i == 1]
            agenttypes_i = agenttypes[batch_idxs == i][agenttopredicts_i == 1][:, 1]
            loc_pred_i = loc_preds[batch_idxs == i][agenttopredicts_i == 1]
            feat_psirads_i = feat_psirads[batch_idxs == i][agenttopredicts_i == 1] 
            ctrs_i = ctrs[batch_idxs == i][agenttopredicts_i == 1]
            track_ids_i = track_ids[batch_idxs == i][agenttopredicts_i == 1]

            assert np.all(agenttypes_i == 1)

            heading_present = feat_psirads_i.reshape((-1, 10))[:, 9]
            # convert heading back into world coordinates
            for actor in range(len(heading_present)):
                heading_present[actor] = heading_present[actor] - thetas_i[actor]
                
                # angle now between -pi and 2pi
                if heading_present[actor] < -1 * (math.pi):
                    heading_present[actor] = heading_present[actor] % (2 * math.pi)
                # if between pi and 2pi
                if heading_present[actor] > math.pi:
                    heading_present[actor] = -1 * ((2 * math.pi) - heading_present[actor])

            psirad_pred_i = get_psirad_pred(loc_pred_i, heading_present, ctrs_i)
            agenttopredicts_i = agenttopredicts_i[agenttopredicts_i == 1]

            rows = []
            for actor in range(int(agenttopredicts_i.sum())):
                for step in range(self.prediction_steps):
                    case_id_row = float(int(case_id)) 
                    track_id_row = int(track_ids_i[actor])
                    frame_id_row = step + 11
                    timestamp_ms_row = int(frame_id_row * 100)
                    agent_type_row = 'car'
                    track_to_predict_row = float(agenttopredicts_i[actor])
                    interesting_agent_row = float(interestingagents_i[actor])
                    x1_row = loc_pred_i[actor, step, 0, 0]
                    y1_row = loc_pred_i[actor, step, 0, 1]
                    psi_rad1_row = psirad_pred_i[actor, step, 0]
                    x2_row = loc_pred_i[actor, step, 1, 0]
                    y2_row = loc_pred_i[actor, step, 1, 1]
                    psi_rad2_row = psirad_pred_i[actor, step, 1]
                    x3_row = loc_pred_i[actor, step, 2, 0]
                    y3_row = loc_pred_i[actor, step, 2, 1]
                    psi_rad3_row = psirad_pred_i[actor, step, 2]
                    x4_row = loc_pred_i[actor, step, 3, 0]
                    y4_row = loc_pred_i[actor, step, 3, 1]
                    psi_rad4_row = psirad_pred_i[actor, step, 3]
                    x5_row = loc_pred_i[actor, step, 4, 0]
                    y5_row = loc_pred_i[actor, step, 4, 1]
                    psi_rad5_row = psirad_pred_i[actor, step, 4]
                    x6_row = loc_pred_i[actor, step, 5, 0]
                    y6_row = loc_pred_i[actor, step, 5, 1]
                    psi_rad6_row = psirad_pred_i[actor, step, 5]

                    rows.append((case_id_row, track_id_row, frame_id_row, timestamp_ms_row, agent_type_row, track_to_predict_row, interesting_agent_row, x1_row, y1_row, psi_rad1_row, x2_row, y2_row, psi_rad2_row, x3_row, y3_row, psi_rad3_row, x4_row, y4_row, psi_rad4_row, x5_row, y5_row, psi_rad5_row, x6_row, y6_row, psi_rad6_row))
            
            file_path = os.path.join(test_set_submission_dir, "{}_sub.csv".format(city))
            if os.path.isfile(file_path):
                # append to existing csv file 
                with open(file_path, "a") as res:
                    wtr = csv.writer(res)
                    for row in rows:
                        wtr.writerow(row)
            else:
                # open new csv file and write to csv file    
                with open(file_path, "w") as res:
                    wtr = csv.writer(res)
                    wtr.writerow(header)
                    for row in rows:
                        wtr.writerow(row)
   
    def init_dgl_graph(self, batch_idxs, ctrs, orig, rot, agenttypes, world_locs):        
        n_scenarios = len(np.unique(batch_idxs))
        graphs, labels = [], []
        for ii in range(n_scenarios):
            label = None

            # number of agents in the scene (currently > 0)
            si = ctrs[batch_idxs == ii].shape[0]
            assert si > 0

            # start with a fully-connected graph
            if si > 1:
                off_diag = np.ones([si, si]) - np.eye(si)
                rel_src = np.where(off_diag)[0]
                rel_dst = np.where(off_diag)[1]

                graph = dgl.graph((rel_src, rel_dst))
            else:
                graph = dgl.graph(([], []), num_nodes=si)

            # separate graph for each scenario
            graph.ndata["ctrs"] = ctrs[batch_idxs == ii]
            graph.ndata["rot"] = rot[batch_idxs == ii]
            graph.ndata["orig"] = orig[batch_idxs == ii]
            graph.ndata["agenttypes"] = agenttypes[batch_idxs == ii].float()
            
            graphs.append(graph)
            labels.append(label)
        
        graphs = dgl.batch(graphs)
        return graphs

    def build_stage_1_graph(self, graph, x, agenttypes, actor_idcs, actor_ctrs, lane_graph):
        all_edges = [x.unsqueeze(1) for x in graph.edges('uv')]
        all_edges = torch.cat(all_edges, 1)
        
        stage_1_graph = dgl.graph((all_edges[:, 0], all_edges[:, 1]), num_nodes = graph.num_nodes())
        stage_1_graph.ndata["ctrs"] = graph.ndata["ctrs"]
        stage_1_graph.ndata["rot"] = graph.ndata["rot"]
        stage_1_graph.ndata["orig"] = graph.ndata["orig"]
        stage_1_graph.ndata["agenttypes"] = graph.ndata["agenttypes"].float()

        stage_1_graph = self.pretrained_relation_header.feature_encoder(stage_1_graph, x, agenttypes, actor_idcs, actor_ctrs, lane_graph)

        return stage_1_graph

    def forward(self, scene_idxs, graph, stage_1_graph, batch_idxs, actor_ctrs, prop_ground_truth = 0.):
        
        if (self.two_stage_training and self.training_stage == 2):
            prh_logits = self.pretrained_relation_header.relation_header(stage_1_graph)
        else:
            prh_logits = torch.ones(graph.num_edges(), 3).to(graph.ndata["xt_enc"].device)
            prh_logits[:, 1:] = 0

        graph.edata["edge_logits"] = prh_logits
        
        all_edges = [x.unsqueeze(1) for x in graph.edges('all')]
        all_edges = torch.cat(all_edges, 1)
        # remove half of the edges (effectively now an undirected graph)
        eids_remove = all_edges[torch.where(all_edges[:, 0] > all_edges[:, 1])[0], 2]
        graph.remove_edges(eids_remove)

        if self.learned_relation_header or (self.two_stage_training and self.training_stage == 2):
            edge_logits = graph.edata.pop("edge_logits")
            edge_probs = my_softmax(edge_logits, -1)

        graph.edata["edge_probs"] = edge_probs

        dag_graph = build_dag_graph_test(graph, self.config)
        
        # only prune graph if we are using the DAGNN
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            dag_graph = prune_graph_johnson(dag_graph)
        
        if self.proposal_header:
            dag_graph, proposals = self.proposal_decoder(dag_graph, actor_ctrs)
        
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            loc_pred = self.trajectory_decoder(dag_graph, prop_ground_truth, batch_idxs)
        
        # loc_pred: shape [N, prediction_steps, num_joint_modes, 2]
        res = {}

        if self.proposal_header:
            res["proposals"] = proposals # trajectory proposal future coordinates
        
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            res["loc_pred"] = loc_pred # predicted future coordinates  
        
        return res

    def load_relation_header(self):
        # load best model from pt file
        path = self.log_path / "best_model_relation_header.pt"
        state = torch.load(path, map_location=dev)
        self.load_state_dict(state['state_dict'])

    def prepare_for_stage_2(self, pretrained_relation_header):
        # first, load model from stage 1 and set weights for stage 2
        path = self.log_path / "best_model_relation_header.pt"
        state = torch.load(path, map_location=dev)
        pretrained_relation_header.load_state_dict(state['state_dict'])

        # second, freeze the weights of the network trained in stage 1
        for param in pretrained_relation_header.parameters():
            param.requires_grad = False

        self.pretrained_relation_header = pretrained_relation_header

    def load_for_eval(self):
        # load best model from pt file
        path = self.log_path / "best_model.pt"
        state = torch.load(path, map_location=dev)
        self.load_state_dict(state['state_dict'])

if __name__ == '__main__':
    config = {}
    config["mode"] = args.mode 
    config["dataset"] = args.dataset 
    config["config_name"] = args.config_name 
    config["num_edge_types"] = args.num_edge_types
    config["h_dim"] = args.h_dim 
    config["num_joint_modes"] = args.num_joint_modes
    config["num_proposals"] = args.num_proposals
    config["max_epochs"] = args.max_epochs 
    config["log_path"] = Path('./logs') / config["config_name"]
    config["lr"] = args.lr 
    config["decoder"] = args.decoder
    config["num_heads"] = args.num_heads
    config["learned_relation_header"] = args.learned_relation_header
    config["n_mapnet_layers"] = args.n_mapnet_layers 
    config["n_l2a_layers"] = args.n_l2a_layers
    config["n_a2a_layers"] = args.n_a2a_layers
    config["resume_training"] = args.resume_training
    config["proposal_coef"] = args.proposal_coef
    config["rel_coef"] = args.rel_coef
    config["proposal_header"] = args.proposal_header
    config["two_stage_training"] = args.two_stage_training
    config["training_stage"] = args.training_stage
    config["ig"] = args.ig
    config["focal_loss"] = args.focal_loss 
    config["gamma"] = args.gamma
    config["weight_0"] = args.weight_0
    config["weight_1"] = args.weight_1
    config["weight_2"] = args.weight_2
    config["teacher_forcing"] = args.teacher_forcing
    config["scheduled_sampling"] = args.scheduled_sampling 
    config["eval_training"] = args.eval_training
    config["supervise_vehicles"] = args.supervise_vehicles
    config["no_agenttype_encoder"] = args.no_agenttype_encoder 
    config["train_all"] = args.train_all

    config["log_path"].mkdir(exist_ok=True, parents=True)
    log = os.path.join(config["log_path"], "log")
    # write stdout to log file
    sys.stdout = Logger(log)

    if args.dataset == 'interaction':
        if config["train_all"]:
            config["num_train_samples"] = 47584 + 11794
        else:
            config["num_train_samples"] = 47584
        config["num_val_samples"] = 11794
        config["num_test_samples"] = 2644
        config["switch_lr_1"] = 40
        config["switch_lr_2"] = 48
        config["lr_step"] = 1/5
        config["input_size"] = 5
        config["prediction_steps"] = 30 
        config["observation_steps"] = 10
        # two agent types: "car", and "pedestrian/bicyclist"
        config["num_agenttypes"] = 2
        config['dataset_path'] = 'dataset_INTERACTION'
        config['tracks_train_reformatted'] = os.path.join(config['dataset_path'], 'train_reformatted')
        config['tracks_val_reformatted'] = os.path.join(config['dataset_path'], 'val_reformatted')
        config['tracks_test_reformatted'] = os.path.join(config['dataset_path'], 'test_reformatted')
        config['num_scales'] = 4
        config["map2actor_dist"] = 20.0
        config["actor2actor_dist"] = 100.0
        config['maps'] = os.path.join(config['dataset_path'], 'maps')
        config['cross_dist'] = 10
        config['cross_angle'] = 1 * np.pi
        config["preprocess"] = True
        config["val_workers"] = 0
        config["workers"] = 0
        if config["train_all"]:
            config["preprocess_train"] = os.path.join(config['dataset_path'], 'preprocess', 'train_all_interaction')
        else:
            config["preprocess_train"] = os.path.join(config['dataset_path'], 'preprocess', 'train_interaction')
        config["preprocess_val"] = os.path.join(config['dataset_path'], 'preprocess', 'val_interaction')
        config["preprocess_test"] = os.path.join(config['dataset_path'], 'preprocess', 'test_interaction')
        config['batch_size'] = args.batch_size

        dataset = InteractionTestDataset(config)  
        print("Loaded preprocessed test data.")  
        val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
        val_loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            num_workers=config["val_workers"],
            sampler=val_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    # Run evaluation code
    if args.mode == 'eval':
        model = FJMP(config)
        m = sum(p.numel() for p in model.parameters())
        print_("Model: {} parameters".format(m))
        print_("Evaluating model...")

        # load model from stage 1 and freeze weights
        if model.two_stage_training and model.training_stage == 2:
            with open(os.path.join(config["log_path"], "config_stage_1.pkl"), "rb") as f:
                config_stage_1 = pickle.load(f) 

            config_stage_1["num_test_samples"] = 2644
            pretrained_relation_header = FJMP(config_stage_1)
            model.prepare_for_stage_2(pretrained_relation_header)
        
        model.load_for_eval()
        # evaluate model
        results = model._eval(val_loader, 1)
        print_("Finished predicting test set.")  