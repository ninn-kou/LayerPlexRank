# LayerPlexRank

```
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

## Files

- **Codes**
  - `LayerPlexRank.py`: Main algorithms and their helper functions.
  - `ExperimentHelpers.py`: Drawing figures of experiments. Modify this file to customize figure settings. 
  - `DataSplitLOOCV.py`: A Python script to generate sub-datasets used in LOOCV experiments.
- **Jupyter Notebooks**
  - `LayerPlexRank - EUAir Experiments.ipynb`: Main experiments. Read this document for experiment details.
  - `LayerPlexRank - Other Experiments (Plotting Figures).ipynb`: Notebook for experiments on other three datasets.
- **Datasets**
  - Collect from <https://manliodedomenico.com/data.php>.
    - EUAir
    - C.Elegans
    - CSAarhus
    - Candida
