from dgl.convert import graph
import dgl
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import dgl.function as fn
import torch
from fractions import gcd
from fjmp_utils import *
from lanegcn_modules import *

class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, ng=32):
        super(LinearRes, self).__init__()

        self.linear1 = nn.Linear(n_in, n_out)
        self.linear2 = nn.Linear(n_out, n_out)
        self.transform_linear = nn.Linear(n_in, n_out)
        self.elu = nn.ELU(inplace=True)
        self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
        self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        self.transform_norm = nn.GroupNorm(gcd(ng, n_out), n_out)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.elu(out)
        out = self.linear2(out)
        out = self.norm2(out)
        
        x = self.transform_linear(x)
        x = self.transform_norm(x)
        
        out  = out + x
        out = self.elu(out)
        return out

class Linear1(nn.Module):
    def __init__(self, n_in, n_out, ng=32):
        super(Linear1, self).__init__()
        self.linear1 = nn.Linear(n_in, n_out)
        self.elu = nn.ELU(inplace=True)
        self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.elu(out).transpose(1,2)
        out = self.norm1(out).transpose(1,2)
        return out

class MLP1(nn.Module):
    def __init__(self, n_in, n_out, ng=32):
        super(MLP1, self).__init__()
        self.linear1 = nn.Linear(n_in, n_out)
        self.linear2 = nn.Linear(n_out, n_out)
        self.elu = nn.ELU(inplace=True)
        self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
        self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.elu(out).transpose(1,2)
        out = self.norm1(out).transpose(1,2)
        out = self.linear2(out)
        out = self.elu(out).transpose(1,2)
        out = self.norm2(out).transpose(1,2)
        return out
    
# 2-layer ELU net with residual connection. Used in GNN layers
class MLP(nn.Module):
    def __init__(self, n_in, n_out, ng=32):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(n_in, n_out)
        self.linear2 = nn.Linear(n_out, n_out)
        self.elu = nn.ELU(inplace=True)
        self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
        self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)

        self.init_weights()

    def forward(self, x):
        out = self.linear1(x)
        out = self.elu(out)
        out = self.norm1(out)
        out = self.linear2(out)
        out = self.elu(out)
        out = self.norm2(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias, 0.1)

# Use LaneGCN backbone L2A --> A2A to build feature representation
class FJMPFeatureEncoder(nn.Module):
    def __init__(self, config):
        super(FJMPFeatureEncoder, self).__init__()
        self.config = config 
        self.h_dim_gru = 256

        if not self.config["no_agenttype_encoder"]:
            self.agenttype_enc = Linear(self.config["num_agenttypes"], self.h_dim_gru)
        self.feat_enc = nn.GRU(self.config['input_size'], self.h_dim_gru, 1)
        self.map_net = MapNet(self.config)
        self.l2a = L2A(self.config)
        self.a2a = A2A(self.config)
        self.downsample = nn.Linear(self.h_dim_gru, self.config['h_dim'])

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, graph, x, agenttypes, actor_idcs, actor_ctrs, lane_graph):
        # [N, T, 5] --> [T, N, 5]
        x = x.transpose(1, 0).to(graph.ndata["ctrs"].device)
        h0 = torch.zeros(1, x.shape[1], self.h_dim_gru).to(x.device)
        # x now contains the final hidden states of the GRU
        _, out = self.feat_enc(x, h0)
        out = out[0]
        
        # additive learnable embedding for agent type
        if not self.config["no_agenttype_encoder"]:
            agenttype_enc = self.agenttype_enc(agenttypes.float().to(x.device))
            out = self.downsample(out + agenttype_enc)
        else:
            out = self.downsample(out)

        # get feature embeddings for each lane node
        nodes, node_idcs, node_ctrs = self.map_net(lane_graph)

        # L2A mechanism
        out = self.l2a(out, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        # A2A mechanism
        out = self.a2a(out, actor_idcs, actor_ctrs)

        # store encodings in the nodes (output of GRU)
        graph.ndata['xt_enc'] = out

        return graph

class FJMPRelationHeader(nn.Module):
    def __init__(self, config):
        super(FJMPRelationHeader, self).__init__()
        self.config = config
        
        self.agenttype_enc = Linear(2 * self.config["num_agenttypes"], self.config["h_dim"])
        self.dist = Linear(2, self.config["h_dim"])
        # convert src/dst features to edge features
        self.f_1e = MLP(self.config['h_dim'] * 4, self.config['h_dim'])

        self.h_1e_out = nn.Sequential(
                LinearRes(self.config['h_dim'], self.config['h_dim']),
                nn.Linear(self.config['h_dim'], self.config["num_edge_types"]),
            )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias, 0.1)
    
    def message_func1(self, edges):
        agenttype_enc = self.agenttype_enc(torch.cat([edges.src['agenttypes'], edges.dst['agenttypes']], dim=-1))
        dist = self.dist(edges.dst['ctrs'] - edges.src['ctrs'])
        h_1e = self.f_1e(torch.cat([edges.src['xt_enc'], edges.dst['xt_enc'], dist, agenttype_enc], dim=-1))
        
        edges.data['h_1e'] = h_1e
        return {'h_1e': h_1e}
    
    def node_to_edge(self, graph):
        # propagate edge features back to nodes
        graph.apply_edges(self.message_func1)

        return graph

    def forward(self, graph):
        graph = self.node_to_edge(graph)
        h_1e_out = self.h_1e_out(graph.edata["h_1e"])
        
        return h_1e_out

# Proposal decoder
class FJMPTrajectoryProposalDecoder(nn.Module):
    def __init__(self, config):
        super(FJMPTrajectoryProposalDecoder, self).__init__()
        self.config = config

        self.f_proposal_1 = LinearRes(self.config['h_dim'] + self.config["num_proposals"], self.config['h_dim'])
        self.f_proposal_2 = nn.Linear(self.config['h_dim'], 2 * config["prediction_steps"])

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias, 0.1)

    def forward(self, graph, actor_ctrs):
        # [N, num_proposals, h_dim]
        graph.ndata["v_n"] = torch.stack([graph.ndata["xt_enc"]] * self.config["num_proposals"], dim=2).transpose(1,2)
                   
        mode_idx = torch.zeros(graph.ndata["v_n"].shape[0], self.config["num_proposals"], self.config["num_proposals"]).to(graph.ndata["v_n"].device)
        for mode in range(self.config["num_proposals"]):
            mode_idx[:, mode, mode] = 1.
        out = torch.cat([graph.ndata["v_n"], mode_idx], dim=-1)
        proposals = []
        for i in range(self.config["num_proposals"]):
            embedding = self.f_proposal_1(out[:, i, :])
            proposal = self.f_proposal_2(embedding)
            proposals.append(proposal)           

        proposals_coordinate_embedding = torch.cat([x.unsqueeze(1) for x in proposals], 1).reshape(-1, self.config["num_proposals"], self.config["prediction_steps"], 2) + graph.ndata["ctrs"].view(-1, 1, 1, 2)

        # Map back to absolute coordinates
        proposals = torch.matmul(proposals_coordinate_embedding, graph.ndata["rot"].unsqueeze(1)) + graph.ndata["orig"].view(-1, 1, 1, 2)
        proposals = proposals.transpose(1, 2)
        
        return graph, proposals


# This is the DAGNN decoder
class FJMPAttentionTrajectoryDecoder(nn.Module):
    def __init__(self, config):
        super(FJMPAttentionTrajectoryDecoder, self).__init__()
        self.config = config

        fc1s = []
        fc2s = []
        fc3s = []
        
        for i in range(self.config["num_heads"]):
            fc1s.append(nn.Linear(self.config["h_dim"], self.config["h_dim"], bias=False))
            fc2s.append(nn.Linear(self.config["h_dim"], self.config["h_dim"], bias=False))
            fc3s.append(nn.Linear(self.config["h_dim"] * 2, 1, bias=False))

        self.fc1s = nn.ModuleList(fc1s)
        self.fc2s = nn.ModuleList(fc2s)
        self.fc3s = nn.ModuleList(fc3s)

        self.leaky_relu = nn.LeakyReLU(0.2)

        ### NOTE: this is not used...
        self.dist = MLP(2, self.config["h_dim"])

        self.agenttype_enc = Linear(2 * self.config["num_agenttypes"], self.config["h_dim"])
        self.encode_preds = LinearRes(2 * config["prediction_steps"], self.config['h_dim'])
        self.fc_agg = nn.GRUCell(self.config["h_dim"], self.config["h_dim"])
        self.f_out = nn.Sequential(
                LinearRes(self.config['h_dim'] + self.config["num_joint_modes"], self.config['h_dim']),
                nn.Linear(self.config['h_dim'], 2 * config["prediction_steps"]),
            )  

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias, 0.1)

    def message_func(self, edges):
        agenttype_enc = self.agenttype_enc(torch.cat([edges.src['agenttypes'], edges.dst['agenttypes']], dim=-1))
        agenttype_enc = agenttype_enc.view(-1, 1, self.config["h_dim"])
        
        # encode future predictions
        ctrs_reactors = edges.dst['ctrs'].view(-1, 1, 1, 2)
        if self.config["teacher_forcing"] or self.config["scheduled_sampling"]:
            pred_future_relative = edges.src['f_decode'].detach() - ctrs_reactors
            ground_truth_futures = edges.src['ground_truth_futures'].to(ctrs_reactors.device)
            ground_truth_futures_relative = torch.stack([ground_truth_futures] * self.config["num_joint_modes"], dim=2).transpose(1,2) - ctrs_reactors
            pred_future_relative[edges.data['use_ground_truth'] == 1] = ground_truth_futures_relative[edges.data['use_ground_truth'] == 1]
        else:
            pred_future_relative = edges.src['f_decode'].detach() - ctrs_reactors
        
        pred_feats = []
        for i in range(self.config["num_joint_modes"]):
            pred_feats.append(self.encode_preds(pred_future_relative.view(-1, self.config["num_joint_modes"], self.config["prediction_steps"] * 2)[:, i, :]))
        pred_feats = torch.cat([x.unsqueeze(1) for x in pred_feats], 1)
        
        feats = pred_feats + agenttype_enc
        
        z_m = [self.fc1s[i](feats) for i in range(self.config["num_heads"])]
        
        ### NOTE: self.fc2s receives little gradient because often the softmax is over 1 edge, which trivially has 0 gradient (this is the 1 function)
        z_n = [self.fc2s[i](edges.dst['v_n']) for i in range(self.config["num_heads"])]        
        z2 = [torch.cat([z_m[i], z_n[i]], dim=-1) for i in range(self.config["num_heads"])]
        e = [self.leaky_relu(self.fc3s[i](z2[i])) for i in range(self.config["num_heads"])]
        
        return {'e': torch.stack(e, dim=1), 'z': torch.stack(z_m, dim=1)}

    def reduce_func(self, nodes):        
        # multi-head graph attention
        alphas = [F.softmax(nodes.mailbox['e'][:, :, i], dim=1) for i in range(self.config["num_heads"])]
        aggs = [torch.sum(alphas[i] * nodes.mailbox['z'][:, :, i], dim=1) for i in range(self.config["num_heads"])]
        agg = torch.mean(torch.stack(aggs, dim=-1), dim=-1)
        
        v_n = self.fc_agg(agg.reshape(-1, self.config["h_dim"]), nodes.data["v_n"].reshape(-1, self.config["h_dim"]))
        v_n = v_n.reshape(-1, self.config["num_joint_modes"], self.config["h_dim"])

        mode_idx = torch.zeros(v_n.shape[0], self.config["num_joint_modes"], self.config["num_joint_modes"]).to(v_n.device)
        for mode in range(self.config["num_joint_modes"]):
            mode_idx[:, mode, mode] = 1.
        out = torch.cat([v_n, mode_idx], dim=-1)  

        preds = []
        for i in range(self.config["num_joint_modes"]):
            preds.append(self.f_out(out[:, i, :]))  

        dests = torch.cat([x.unsqueeze(1) for x in preds], 1)
        f_decode = dests.reshape(-1, self.config["num_joint_modes"], self.config["prediction_steps"], 2) + nodes.data["ctrs"].view(-1, 1, 1, 2)      

        return {'v_n': v_n, 'f_decode': f_decode}         

    def apply_source_nodes(self, graph):
        # first retrieve the topological order
        top_order = dgl.topological_nodes_generator(graph)
        source_node_idxs = top_order[0] 

        # 0s are the message for the root nodes
        inp = graph.ndata["v_n"][source_node_idxs].reshape(-1, self.config["h_dim"])
        v_n = self.fc_agg(torch.zeros_like(inp).to(inp.device), inp)
        v_n = v_n.reshape(-1, self.config["num_joint_modes"], self.config["h_dim"])

        graph.ndata["v_n"][source_node_idxs] = v_n

        mode_idx = torch.zeros(source_node_idxs.shape[0], self.config["num_joint_modes"], self.config["num_joint_modes"]).to(graph.ndata["v_n"].device)
        for mode in range(self.config["num_joint_modes"]):
            mode_idx[:, mode, mode] = 1.
        out = torch.cat([v_n, mode_idx], dim=-1)
        preds = []
        for i in range(self.config["num_joint_modes"]):
            preds.append(self.f_out(out[:, i, :]))

        dests = torch.cat([x.unsqueeze(1) for x in preds], 1)
        f_decode = dests.reshape(-1, self.config["num_joint_modes"], self.config["prediction_steps"], 2) + graph.ndata["ctrs"][source_node_idxs].view(-1, 1, 1, 2)
        graph.ndata["f_decode"][source_node_idxs] = f_decode

        return graph

    def forward(self, graph, prop_ground_truth=0., batch_idcs=None):
        
        graph.ndata["xt_dec"] = graph.ndata["xt_enc"]
        # [N, h_dim]
        graph.ndata["v_n"] = torch.stack([graph.ndata["xt_dec"]] * self.config["num_joint_modes"], dim=2).transpose(1,2)

        # dgl does not like empty graphs
        if len(graph.edges('eid')) != 0:
            # This contains our predictions
            # [N, best_N, prediction_steps, 2]
            graph.ndata["f_decode"] = torch.zeros(graph.ndata["xt_enc"].shape[0], self.config["num_joint_modes"], self.config["prediction_steps"], 2).to(graph.ndata["v_n"].device)
            
            # First apply function to source nodes
            graph = self.apply_source_nodes(graph)        
            
            if self.config["teacher_forcing"] or self.config["scheduled_sampling"]:
                # Here we set up an edata, where 0 corresponds to using parent's future prediction and 1 corresponds to using parent's ground-truth future
                weights = torch.Tensor([1-prop_ground_truth, prop_ground_truth]).to(graph.ndata["v_n"].device)
                prob_vector = torch.multinomial(weights, graph.num_edges(), replacement=True)
                graph.edata["use_ground_truth"] = prob_vector
            
            # propagate information according to partial order of DAG
            dgl.prop_nodes_topo(graph,
                                message_func=self.message_func,
                                reduce_func=self.reduce_func)

            f_decode = graph.ndata.pop("f_decode")
            pred_loc = torch.matmul(f_decode, graph.ndata["rot"].unsqueeze(1)) + graph.ndata["orig"].view(-1, 1, 1, 2)
            pred_loc = pred_loc.transpose(1, 2)
        
        # special case when we have an empty dgl graph
        else:
            v_n = graph.ndata.pop("v_n") 
            mode_idx = torch.zeros(v_n.shape[0], self.config["num_joint_modes"], self.config["num_joint_modes"]).to(v_n.device)
            for mode in range(self.config["num_joint_modes"]):
                mode_idx[:, mode, mode] = 1.
            out = torch.cat([v_n, mode_idx], dim=-1)  

            preds = []
            for i in range(self.config["num_joint_modes"]):
                preds.append(self.f_out(out[:, i, :]))      
            
            dests = torch.cat([x.unsqueeze(1) for x in preds], 1)
            
            f_decode = dests.reshape(-1, self.config["num_joint_modes"], self.config["prediction_steps"], 2) + graph.ndata["ctrs"].view(-1, 1, 1, 2)
            
            pred_loc = torch.matmul(f_decode, graph.ndata["rot"].unsqueeze(1)) + graph.ndata["orig"].view(-1, 1, 1, 2)
            pred_loc = pred_loc.transpose(1, 2)
        
        # [N, prediction_steps, num_joint_modes, 2]
        return pred_loc

class LaneGCNHeader(nn.Module):
    def __init__(self, config):
        super(LaneGCNHeader, self).__init__()
        self.config = config
        self.f_out = nn.Sequential(
                    LinearRes(self.config['h_dim'] + self.config["num_joint_modes"], self.config['h_dim']),
                    nn.Linear(self.config['h_dim'], 2 * config["prediction_steps"]),
                )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias, 0.1)
    
    def forward(self, graph, prop_ground_truth=None, batch_idcs=None):
        # [N, num_joint_modes, h_dim]
        out = torch.stack([graph.ndata['xt_enc']] * self.config["num_joint_modes"], dim=2).transpose(1,2)
        mode_idx = torch.zeros(graph.ndata['xt_enc'].shape[0], self.config["num_joint_modes"], self.config["num_joint_modes"]).to(out.device)
        for mode in range(self.config["num_joint_modes"]):
            mode_idx[:, mode, mode] = 1.
        out = torch.cat([out, mode_idx], dim=-1)
        preds = []
        for i in range(self.config["num_joint_modes"]):
            preds.append(self.f_out(out[:, i, :]))
        
        graph.ndata["out"] = torch.cat([x.unsqueeze(1) for x in preds], 1).reshape(-1, self.config["num_joint_modes"], self.config["prediction_steps"], 2)
        
        # [N, num_joint_modes, 2 * prediction_steps] --> [N, num_joint_modes, prediction_steps, 2]
        pred_loc = graph.ndata["out"].reshape(-1, self.config["num_joint_modes"], self.config["prediction_steps"], 2)

        # Translate pred_loc by the object centers (this is for normalization)
        # [N, num_joint_modes, prediction_steps, 2], add across all samples and timesteps
        pred_loc = pred_loc + graph.ndata["ctrs"].view(-1, 1, 1, 2)
        
        # map back to world coordinates
        pred_loc = torch.matmul(pred_loc, graph.ndata["rot"].unsqueeze(1)) + graph.ndata["orig"].view(-1, 1, 1, 2)
        pred_loc = pred_loc.transpose(1, 2)
        
        # [N, num_joint_modes, prediction_steps, 2] --> [N, prediction_steps, num_joint_modes, 2]
        return pred_loc