<div align="center">
<h1>MapTR <img src="assets/map.png" width="30"></h1>
<h3>Structured Modeling and Learning for Online Vectorized HD Map Construction</h3>

[Bencheng Liao](https://github.com/LegendBC)<sup>1,2,3</sup> \*, [Shaoyu Chen](https://scholar.google.com/citations?user=PIeNN2gAAAAJ&hl=en&oi=sra)<sup>1,3</sup> \*, [Xinggang Wang](https://xinggangw.info/)<sup>1 :email:</sup>, [Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ&hl=zh-CN)<sup>1,3</sup>, [Qian Zhang](https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN)<sup>3</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, [Chang Huang](https://scholar.google.com/citations?user=IyyEKyIAAAAJ&hl=zh-CN)<sup>3</sup>
 
<sup>1</sup> School of EIC, HUST, <sup>2</sup> Institute of Artificial Intelligence, HUST, <sup>3</sup> Horizon Robotics

(\*) equal contribution, (<sup>:email:</sup>) corresponding author.

ArXiv Preprint ([arXiv 2208.14437](https://arxiv.org/abs/2208.14437))

[openreview ICLR'23](https://openreview.net/forum?id=k7p_YAO7yE), accepted as **ICLR Spotlight**

</div>

#
### News
* **`Aug. 14th, 2023`:** As required by many researchers, the code of MapTR-based map annotation framework (VMA) will be released at https://github.com/hustvl/VMA recently.
* **`Aug. 10th, 2023`:** We release [MapTRv2](https://arxiv.org/abs/2308.05736) on Arxiv. MapTRv2 demonstrates much stronger performance and much faster convergence. To better meet the requirement of the downstream planner (like [PDM](https://github.com/autonomousvision/nuplan_garage)), we introduce an extra semantic——centerline (using path-wise modeling proposed by [LaneGAP](https://github.com/hustvl/LaneGAP)). Code & model will be released in late August. Please stay tuned!
* **`May. 12th, 2023`:** MapTR now support various bevencoder, such as [BEVFormer encoder](projects/configs/maptr/maptr_tiny_r50_24e_bevformer.py) and [BEVFusion bevpool](projects\configs\maptr\maptr_tiny_r50_24e_bevpool.py). Check it out!
* **`Apr. 20th, 2023`:** Extending MapTR to a general map annotation framework ([paper](https://arxiv.org/pdf/2304.09807.pdf), [code](https://github.com/hustvl/VMA)), with high flexibility in terms of spatial scale and element type.
* **`Mar. 22nd, 2023`:** By leveraging MapTR, VAD ([paper](https://arxiv.org/abs/2303.12077), [code](https://github.com/hustvl/VAD))  models the driving scene as fully vectorized representation, achieving SoTA end-to-end planning performance!
* **`Jan. 21st, 2023`:** MapTR is accepted to ICLR 2023 as **Spotlight Presentation**!
* **`Nov. 11st, 2022`:** We release an initial version of MapTR.
* **`Aug. 31st, 2022`:** We released our paper on Arxiv. Code/Models are coming soon. Please stay tuned! ☕️


## Introduction
<div align="center"><h4>MapTR is a simple, fast and strong online vectorized HD map construction framework.</h4></div>

![framework](assets/teaser.png "framework")

We present MapTR, a structured  end-to-end framework for efficient online vectorized HD map construction. 
We propose a unified  permutation-based modeling approach,
*ie*, modeling map element as a point set with a group of equivalent permutations, which avoids the definition ambiguity of map element and eases learning.
We adopt a hierarchical query embedding scheme to flexibly encode structured map information and perform hierarchical bipartite matching for map element learning. MapTR achieves the best performance and efficiency among existing vectorized map construction approaches on nuScenes dataset. In particular, MapTR-nano runs at real-time inference speed ( $25.1$ FPS ) on RTX 3090, $8\times$ faster than the existing state-of-the-art camera-based method while achieving $3.3$ higher mAP.
MapTR-tiny significantly outperforms the existing state-of-the-art multi-modality method by $13.5$ mAP while being faster.
Qualitative results show that MapTR maintains stable and robust map construction quality in complex and various driving scenes. MapTR is of great application value in autonomous driving. 

## Models
> Results from the [paper](https://arxiv.org/abs/2308.05736)


| Method | Backbone | Lr Schd | mAP| FPS|
| :---: | :---: |  :---: | :---: | :---:|
| MapTR | R18 | 110ep | 45.9 | 35.0| 
| MapTR | R50 | 24ep | 50.3 | 15.1| 
| MapTR | R50 | 110ep | 58.7|15.1| 
| MapTRv2 | R18 |  110ep | 44.2 | 25.1| 
| MapTRv2 | R50 | 24ep | 50.3 | 11.2| 
| MapTRv2 | R50 | 110ep | 58.7|11.2| 

**Notes**: 

- FPS is measured on NVIDIA RTX3090 GPU with batch size of 1 (containing 6 view images).
- All the experiments are performed on 8 NVIDIA GeForce RTX 3090 GPUs. 

> Results from this repo.

## Qualitative results on nuScenes val set and Argoverse2 val set
<div align="center"><h4>MapTR maintains stable and robust map construction quality in various driving scenes.</h4></div>


### *End-to-end Planning*
https://user-images.githubusercontent.com/26790424/229679664-0e9ba5e8-bf2c-45e0-abbc-36d840ee5cc9.mp4



## Getting Started
- [Installation](docs/install.md)
- [Prepare Dataset](docs/prepare_dataset.md)
- [Train and Eval](docs/train_eval.md)
- [Visualization](docs/visualization.md)


## Catalog

- [ ] centerline detection & topology support
- [x] multi-modal checkpoints
- [x] multi-modal code
- [ ] lidar modality code
- [x] argoverse2 dataset 
- [x] Nuscenes dataset 
- [x] MapTR checkpoints
- [x] MapTR code
- [x] Initialization

## Acknowledgements

MapTR is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). It is also greatly inspired by the following outstanding contributions to the open-source community: [BEVFusion](https://github.com/mit-han-lab/bevfusion), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [HDMapNet](https://github.com/Tsinghua-MARS-Lab/HDMapNet), [GKT](https://github.com/hustvl/GKT), [VectorMapNet](https://github.com/Mrmoore98/VectorMapNet_code).

## Citation
If you find MapTR is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```bibtex
@inproceedings{MapTR,
  title={MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction},
  author={Liao, Bencheng and Chen, Shaoyu and Wang, Xinggang and Cheng, Tianheng, and Zhang, Qian and Liu, Wenyu and Huang, Chang},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
```bibtex
@inproceedings{MapTRv2,
  title={MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction},
  author={Liao, Bencheng and Chen, Shaoyu and Zhang, Yunchi and Jiang, Bo and Zhang, Qian and Liu, Wenyu and Huang, Chang and Wang, Xinggang},
  booktitle={arXiv preprint arXiv: 2308.05736},
  year={2023}
}
```
