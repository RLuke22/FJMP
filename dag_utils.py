import numpy, torch, dgl
import networkx as nx 
import matplotlib.pyplot as plt
from tqdm import tqdm

def build_dag_graph(graph, config):
    edge_type = torch.argmax(graph.edata["edge_probs"], dim=1)
    all_edges = [x.unsqueeze(1) for x in graph.edges('uv')]
    all_edges = torch.cat(all_edges, 1)

    # i --> j (i < j) edges in the graph
    src_edges_type_1 = all_edges[edge_type == 1][:, 0]
    dest_edges_type_1 = all_edges[edge_type == 1][:, 1]

    # j --> i (i < j) edges in the graph
    src_edges_type_2 = all_edges[edge_type == 2][:, 1]
    dest_edges_type_2 = all_edges[edge_type == 2][:, 0]

    dag_graph = dgl.graph((torch.cat([src_edges_type_1, src_edges_type_2], dim=0), torch.cat([dest_edges_type_1, dest_edges_type_2], dim=0)), num_nodes = graph.num_nodes())
    dag_edge_probs = torch.cat([graph.edata["edge_probs"][edge_type == 1][:, 1], graph.edata["edge_probs"][edge_type == 2][:, 2]], dim=0)
    dag_graph.edata["edge_probs"] = dag_edge_probs

    # Transfer features into "dagified" graph
    dag_graph.ndata["xt_enc"] = graph.ndata["xt_enc"] 
    dag_graph.ndata["ctrs"] = graph.ndata["ctrs"]
    dag_graph.ndata["rot"] = graph.ndata["rot"]
    dag_graph.ndata["orig"] = graph.ndata["orig"]
    dag_graph.ndata["agenttypes"] = graph.ndata["agenttypes"].float()
    dag_graph.ndata["ground_truth_futures"] = graph.ndata["ground_truth_futures"].float()
    dag_graph.ndata["has_preds"] = graph.ndata["has_preds"].float()

    return dag_graph

def prune_graph_johnson(dag_graph):
    """
    dag_graph: DGL graph with weighted edges
    graph contains edge property "edge_probs" which contains predicted probability of each edge type 

    Based on the predicted probabilities, prune graph until it is a DAG based on Johnson's algorithm

    Note that we can think of a batch of graphs as one big graph and apply the pruning procedure on the entire batch at once.
    """

    G = dgl.to_networkx(dag_graph.cpu(), node_attrs=None, edge_attrs=None)
    cycles = nx.simple_cycles(G)

    # First identify cycles in graph
    eids = []
    for cycle in cycles:
        out_cycle = torch.Tensor(cycle).to(dag_graph.device).long()
        in_cycle = torch.roll(out_cycle, 1)

        eids.append(dag_graph.edge_ids(in_cycle, out_cycle))

    to_remove = []
    while len(eids) > 0:
        edge_probs_cycle = dag_graph.edata["edge_probs"][eids[0]]
        remove_eid = eids[0][torch.argmin(edge_probs_cycle)]
        to_remove.append(remove_eid)

        eids.pop(0)
        to_pop = []
        for j, eid_cycle in enumerate(eids):
            if remove_eid in eid_cycle:
                to_pop.append(j)
        
        eids = [v for i, v in enumerate(eids) if i not in to_pop]

    dag_graph.remove_edges(to_remove)

    return dag_graph

def build_dag_graph_test(graph, config):
    edge_type = torch.argmax(graph.edata["edge_probs"], dim=1)
    all_edges = [x.unsqueeze(1) for x in graph.edges('uv')]
    all_edges = torch.cat(all_edges, 1)

    # i --> j (i < j) edges in the graph
    src_edges_type_1 = all_edges[edge_type == 1][:, 0]
    dest_edges_type_1 = all_edges[edge_type == 1][:, 1]

    # j --> i (i < j) edges in the graph
    src_edges_type_2 = all_edges[edge_type == 2][:, 1]
    dest_edges_type_2 = all_edges[edge_type == 2][:, 0]

    dag_graph = dgl.graph((torch.cat([src_edges_type_1, src_edges_type_2], dim=0), torch.cat([dest_edges_type_1, dest_edges_type_2], dim=0)), num_nodes = graph.num_nodes())
    dag_edge_probs = torch.cat([graph.edata["edge_probs"][edge_type == 1][:, 1], graph.edata["edge_probs"][edge_type == 2][:, 2]], dim=0)
    dag_graph.edata["edge_probs"] = dag_edge_probs

    # Transfer features into "dagified" graph
    dag_graph.ndata["xt_enc"] = graph.ndata["xt_enc"] 
    dag_graph.ndata["ctrs"] = graph.ndata["ctrs"]
    dag_graph.ndata["rot"] = graph.ndata["rot"]
    dag_graph.ndata["orig"] = graph.ndata["orig"]
    dag_graph.ndata["agenttypes"] = graph.ndata["agenttypes"].float()

    return dag_graph