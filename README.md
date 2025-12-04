## Physis-Aware Recurrent Convolutional Neural Network
This repository contains implementations of the Physics-Aware Recurrent Convolutional Neural Network (PARCv2) and several of its architectural variants. These models are designed for learning and forecasting spatiotemporal fields governed by partial differential equations, with the core PARCv2 design emphasizing physics-informed inductive biases, a differentiator–integrator decomposition of dynamics, and stable long-horizon rollout in advection- and diffusion-dominated regimes.

### Requirements & Installation
The recommended way to install the models and utilities in this repository is to use the pre-built wheel packages provided under the Releases tab.

To install a released version, download the pre-built wheel pakcages and 
```
pip install multiparc-<version>-py3-none-any.whl 
```
Replace ```<version>``` with the version of package downloaded.

### Tutorial Notebooks
[](https://github.com/chengxinlun/parcv2_trl2d)

### Variants included
* PARCv2
* Multi-resolution PARCv2

### Citation
If you use PARCv2 or any of the model variants provided in this repository, please cite the PARCv2 paper and the corresponding variants paper:
#### PARCv2
```
@inproceedings{nguyen2024parcv2,
  title={PARCv2: Physics-aware recurrent convolutional neural networks for spatiotemporal dynamics modeling},
  author={Nguyen, Phong CH and Cheng, Xinlun and Azarfar, Shahab and Seshadri, Pradeep and Nguyen, Yen T and Kim, Munho and Choi, Sanghun and Udaykumar, HS and Baek, Stephen},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={37649--37666},
  year={2024}
}
@article{cheng2024physics,
  title={Physics-aware recurrent convolutional neural networks for modeling multiphase compressible flows},
  author={Cheng, Xinlun and Nguyen, Phong CH and Seshadri, Pradeep K and Verma, Mayank and Gray, Zo{\"e} J and Beerman, Jack T and Udaykumar, HS and Baek, Stephen S},
  journal={International Journal of Multiphase Flow},
  volume={177},
  pages={104877},
  year={2024},
  publisher={Elsevier}
}
```
#### Multi-resolution PARCv2
```
@article{cheng2025multi,
    author = {Cheng, Xinlun and Choi, Joseph and Udaykumar, H. S. and Baek, Stephen},
    title = {Multi-resolution physics-aware recurrent convolutional neural network for complex flows},
    journal = {APL Machine Learning},
    volume = {3},
    number = {4},
    pages = {046110},
    year = {2025},
    month = {12},
    doi = {10.1063/5.0295883},
}
```
