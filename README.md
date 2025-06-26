<div align="center">

<h2>LayerPlexRank: Exploring Node Centrality and Layer Influence through Algebraic Connectivity in Multiplex Networks</h2>

**33rd ACM International Conference on Information and Knowledge Management (CIKM) 2024**

[Hao Ren](https://ninn-kou.github.io/), [Jiaojiao Jiang](https://www.unsw.edu.au/staff/jiaojiao-jiang)
School of Computer Science and Engineering, University of New South Wales

[![arXiv](https://img.shields.io/badge/arXiv-2405.05576-b31b1b.svg)](https://arxiv.org/abs/2405.05576) [![ACM DL](https://img.shields.io/badge/ACM%20DL-10.1145/3627673.3679950-1d1d1b.svg)](https://dl.acm.org/doi/10.1145/3627673.3679950)

</div>

### Abstract

> As the calculation of centrality in complex networks becomes increasingly vital across technological, biological, and social systems, precise and scalable ranking methods are essential for understanding these networks. This paper introduces LayerPlexRank, an algorithm that simultaneously assesses node centrality and layer influence in multiplex networks using algebraic connectivity metrics. This method enhances the robustness of the ranking algorithm by effectively assessing structural changes across layers using random walk, considering the overall connectivity of the graph. We substantiate the utility of LayerPlexRank with theoretical analyses and empirical validations on varied real-world datasets, contrasting it with established centrality measures.

### Usage

**For usage instructions, demonstrations, and experimental results, please refer to the Jupyter notebook for the [EUAir experiment](https://github.com/ninn-kou/LayerPlexRank/blob/main/LayerPlexRank%20-%20EUAir%20Experiments.ipynb). Notebooks for the other three experiments can also be found at [this link](https://github.com/ninn-kou/LayerPlexRank/blob/main/LayerPlexRank%20-%20Other%20Experiments%20(Plotting%20Figures).ipynb).**

### Jupyter Notebooks

- `LayerPlexRank - EUAir Experiments.ipynb`: Main experiments. Read this document for experiment details.
- `LayerPlexRank - Other Experiments (Plotting Figures).ipynb`: Notebook for experiments on other three datasets.

### Codes

- `LayerPlexRank.py`: Main algorithms and their helper functions.
- `ExperimentHelpers.py`: Drawing figures of experiments. Modify this file to customize figure settings.
- `DataSplitLOOCV.py`: A Python script to generate sub-datasets used in LOOCV (Leave-One-Out Cross-Validation) experiments.

### Datasets

|               | Full Name                       | Layers | Nodes | Edges |
|:--------------|:--------------------------------|:------:|:-----:|:-----:|
| **EUAir**     | EU-Air Transportation Multiplex | 37     | 450   | 3588  |
| **C.Elegans** | C.ELEGANS Multiplex Connectome  | 3      | 279   | 5863  |
| **CSAarhus**  | CS-Aarhus                       | 6      | 61    | 620   |
| **Candida**   | Candida Multiplex GPI Network   | 7      | 367   | 397   |

*Datasets were collected from <https://manliodedomenico.com/data.php>.*

### Citation

```bibtex
@inproceedings{10.1145/3627673.3679950,
  author = {Ren, Hao and Jiang, Jiaojiao},
  title = {LayerPlexRank: Exploring Node Centrality and Layer Influence through Algebraic Connectivity in Multiplex Networks},
  year = {2024},
  isbn = {9798400704369},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3627673.3679950},
  doi = {10.1145/3627673.3679950},
  abstract = {As the calculation of centrality in complex networks becomes increasingly vital across technological, biological, and social systems, precise and scalable ranking methods are essential for understanding these networks. This paper introduces LayerPlexRank, an algorithm that simultaneously assesses node centrality and layer influence in multiplex networks using algebraic connectivity metrics. This method enhances the robustness of the ranking algorithm by effectively assessing structural changes across layers using random walk, considering the overall connectivity of the graph. We substantiate the utility of LayerPlexRank with theoretical analyses and empirical validations on varied real-world datasets, contrasting it with established centrality measures.},
  booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages = {4010–4014},
  numpages = {5},
  keywords = {algebraic connectivity, centrality, influence, multiplex networks},
  location = {Boise, ID, USA},
  series = {CIKM '24}
}
```
