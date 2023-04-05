<br />

TL;DR: We propose **FJMP**, a **F**actorized **J**oint **M**otion **P**rediction framework that efficiently generates joint trajectory predictions for driving scenarios with an arbitrarily large number of agents by factorizing the joint prediction task into a sequence of marginal and conditional predictions. FJMP models the future scene interaction dynamics as a sparse directed interaction graph, where an edge denotes an explicit interaction between a pair of agents, and
the direction of the edge is determined by their influencer-reactor relationship. We propose a mechanism to efficiently prune the interaction graph into a directed acyclic graph (DAG). Joint future trajectory predictions are then decoded as a sequence of marginal and conditional predictions according to the
partial ordering of the DAG. We validate FJMP on the INTERACTION and Argoverse 2 datasets and demonstrate that FJMP produces more accurate and scene-consistent joint trajectory predictions than non-factorized approaches.

**Ranks 1st** on the [INTERACTION Multi-Agent Prediction Benchmark](http://challenge.interaction-dataset.com/leader-board)

![img](src/model.png)

## Abstract

Predicting the future motion of road agents is a critical task in an autonomous driving pipeline. In this work, we address the problem of generating a set of scene-level, or joint, future trajectory predictions in multi-agent driving scenarios. To this end, we propose FJMP, a Factorized Joint Motion Prediction framework for multi-agent interactive driving scenarios. FJMP models the future scene interaction dynamics as a sparse directed interaction graph, where edges denote explicit interactions between agents. We then prune the graph into a directed acyclic graph (DAG) and decompose the joint prediction task into a sequence of marginal and conditional predictions according to the partial ordering of the DAG, where joint future trajectories are decoded using a directed acyclic graph neural network (DAGNN). We conduct experiments on the INTERACTION and Argoverse 2 datasets and demonstrate that FJMP produces more accurate and scene-consistent joint trajectory predictions than non-factorized approaches, especially on the most interactive and kinematically interesting agents. FJMP ranks 1st on the multi-agent test leaderboard of the INTERACTION dataset.

## Examples


## Citation

```bibtex
@InProceedings{rowe2023fjmp,
  title={FJMP: Factorized Joint Multi-Agent Motion Prediction over Learned Directed Acyclic Interaction Graphs},
  author={Rowe, Luke and Ethier, Martin and Dykhne, Eli-Henry and Czarnecki, Krzysztof},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
