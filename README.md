# Transformers are Efficient Hierarchical Chemical Graph Learners

Implementation of Subgraph Transformer (SubFormer) based on torch-geometric package. arXiv: https://arxiv.org/abs/2310.01704

## Abstract
Transformers, adapted from natural language processing, are emerging as a leading approach for graph representation learning. Contemporary graph transformers often treat nodes or edges as separate tokens. This approach leads to computational challenges for even moderately-sized graphs due to the quadratic scaling of self-attention complexity with token count. In this paper, we introduce SubFormer, a graph transformer that operates on subgraphs that aggregate information by a message-passing mechanism. This approach reduces the number of tokens and enhances learning long-range interactions. We demonstrate SubFormer on benchmarks for predicting molecular properties from chemical structures and show that it is competitive with state-of-the-art graph transformers at a fraction of the computational cost, with training times on the order of minutes on a consumer-grade graphics card. We interpret the attention weights in terms of chemical structures. We show that SubFormer exhibits limited over-smoothing and avoids over-squashing, which is prevalent in traditional graph neural networks. 

## Requirements
Install the following packages:
```bash
pip install torch>=1.13
pip install torch_geometric>=2.3
pip install rdkit
pip install ogb
pip install tqdm
pip install networkx
```

## Installation
```bash
pip install -e .
```

## Results

|     Dataset     |  Metric  | Accuracy | # Epoch | Time/Epoch (sec) |
|:---------------:|:--------:|:--------:|:-------:|:----------------:|
|       ZINC      |    MAE   |   0.071  | 1000    | 3                |
|      TOX21      |  ROC-AUC |   0.851  | 40      | 4                |
|     TOXCAST     | ROC-AUC  |   0.752  | 100     | 5                |
|      MOLHIV     |  ROC-AUC |   0.795  | 30      | 45               |
|       MUV       |  PRC-AUC |   0.182  | 20      | 36               |
| Pipetide-Struct |    MAE   |  0.2487  | 100     | 9                |


## Citation
If you use this work in your research, please cite our paper:

```bibtex
@misc{pengmei2023transformers,
      title={Transformers are efficient hierarchical chemical graph learners}, 
      author={Zihan Pengmei and Zimu Li and Chih-chan Tien and Risi Kondor and Aaron R. Dinner},
      year={2023},
      eprint={2310.01704},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Our work has also been inspired by the following papers, so please consider citing them while using our method:

```bibtex
@misc{jin2019junction,
      title={Junction Tree Variational Autoencoder for Molecular Graph Generation}, 
      author={Wengong Jin and Regina Barzilay and Tommi Jaakkola},
      year={2019},
      eprint={1802.04364},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{ying2021transformers,
      title={Do Transformers Really Perform Bad for Graph Representation?}, 
      author={Chengxuan Ying and Tianle Cai and Shengjie Luo and Shuxin Zheng and Guolin Ke and Di He and Yanming Shen and Tie-Yan Liu},
      year={2021},
      eprint={2106.05234},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{fey2020hierarchical,
      title={Hierarchical Inter-Message Passing for Learning on Molecular Graphs}, 
      author={Matthias Fey and Jan-Gin Yuen and Frank Weichert},
      year={2020},
      eprint={2006.12179},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

```
