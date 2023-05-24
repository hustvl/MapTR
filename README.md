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
* **`May. 12th, 2023`:** MapTR now support various bevencoder, such as [BEVFormer encoder](projects/configs/maptr/maptr_tiny_r50_24e_bevformer.py) and [BEVFusion bevpool](projects\configs\maptr\maptr_tiny_r50_24e_bevpool.py). Check it out!
* **`Apr. 20th, 2023`:** Extending MapTR to a general map annotation framework ([paper](https://arxiv.org/pdf/2304.09807.pdf)), with high flexibility in terms of spatial scale and element type.
* **`Mar. 22nd, 2023`:** By leveraging MapTR, VAD ([paper](https://arxiv.org/abs/2303.12077), [code](https://github.com/hustvl/VAD))  models the driving scene as fully vectorized representation, achieving SoTA end-to-end planning performance!
* **`Jan. 21st, 2023`:** MapTR is accepted to ICLR 2023 as **Spotlight Presentation**!
* **`Nov. 11st, 2022`:** We release an initial version of MapTR.
* **`Aug. 31st, 2022`:** We released our paper on Arxiv. Code/Models are coming soon. Please stay tuned! ☕️


## Introduction
<div align="center"><h4>MapTR is a simple, fast and strong online vectorized HD map construction framework.</h4></div>

![framework](assets/framework.png "framework")

We present MapTR, a structured  end-to-end framework for efficient online vectorized HD map construction. 
We propose a unified  permutation-based modeling approach,
*ie*, modeling map element as a point set with a group of equivalent permutations, which avoids the definition ambiguity of map element and eases learning.
We adopt a hierarchical query embedding scheme to flexibly encode structured map information and perform hierarchical bipartite matching for map element learning. MapTR achieves the best performance and efficiency among existing vectorized map construction approaches on nuScenes dataset. In particular, MapTR-nano runs at real-time inference speed ( $25.1$ FPS ) on RTX 3090, $8\times$ faster than the existing state-of-the-art camera-based method while achieving $3.3$ higher mAP.
MapTR-tiny significantly outperforms the existing state-of-the-art multi-modality method by $13.5$ mAP while being faster.
Qualitative results show that MapTR maintains stable and robust map construction quality in complex and various driving scenes. MapTR is of great application value in autonomous driving. 

## Models
> Results from the [paper](https://arxiv.org/abs/2208.14437)


| Method | Backbone | BEVEncoder |Lr Schd | mAP| FPS|memroy | 
| :---: | :---: | :---: | :---: | :---: | :---:|:---:|
| MapTR-nano | R18 | GKT | 110ep | 44.2 | 25.1| 11907M (bs 24) |
| MapTR-tiny | R50 | GKT |24ep | 50.3 | 11.2| 10287M (bs 4) | 
| MapTR-tiny | R50 | GKT |110ep | 58.7|11.2| 10287M (bs 4)|

**Notes**: 

- FPS is measured on NVIDIA RTX3090 GPU with batch size of 1 (containing 6 view images).
- All the experiments are performed on 8 NVIDIA GeForce RTX 3090 GPUs. 

> Results from this repo. FPSs are much higher.

| Method | Backbone | BEVEncoder |Lr Schd | mAP| FPS|memroy | Config | Download |
| :---: | :---: | :---: | :---: |  :---: | :---:|:---:| :---: | :---: |
| MapTR-nano | R18 |GKT | 110ep |46.3  |48.2| 11907M (bs 24) |[config](projects/configs/maptr/maptr_nano_r18_110e.py) |[model](https://drive.google.com/file/d/1-wVO1pZhFif2igJoz-s451swQvPSto2m/view?usp=sharing) / [log](https://drive.google.com/file/d/1Hd25seDQKn8Vv6AQxPfSoiu-tY2i4Haa/view?usp=sharing) |
| MapTR-tiny | R50 | GKT |24ep | 50.0 |18.4| 10287M (bs 4) | [config](projects/configs/maptr/maptr_tiny_r50_24e.py)|[model](https://drive.google.com/file/d/1n1FUFnRqdskvmpLdnsuX_VK6pET19h95/view?usp=share_link) / [log](https://drive.google.com/file/d/1nvPkk0EMHV8Q82E9usEKKYx7P38bCx1U/view?usp=share_link) |
| MapTR-tiny | R50 |GKT | 110ep | 59.3 |18.4| 10287M (bs 4)|[config](projects/configs/maptr/maptr_tiny_r50_110e.py) |[model](https://drive.google.com/file/d/1SCF93LEEmXU0hMwPiUz9p2CWbL1FpB1h/view?usp=share_link) / [log](https://drive.google.com/file/d/1TQ4j_0Sf2ipzeYsEZZAHYzX4dCUaBqyp/view?usp=share_link) |
| MapTR-tiny | Camera & LiDAR | GKT |24ep | 62.7 | 6.0 | 11858M (bs 4)|[config](projects/configs/maptr/maptr_tiny_fusion_24e.py) |[model](https://drive.google.com/file/d/1CFlJrl3ZDj3gIOysf5Cli9bX5LEYSYO4/view?usp=share_link) / [log](https://drive.google.com/file/d/1rb3S4oluxdZjNm2aJ5lBH23jrkYIaJbC/view?usp=share_link) |
| MapTR-tiny | R50 | bevpool |24ep | 50.1 | 17.2 | 9817M (bs 4)|[config](projects/configs/maptr/maptr_tiny_r50_24e_bevpool.py) |[model](https://drive.google.com/file/d/16PK9XohV55_3qPVDtpXIl4_Iumw9EnfA/view?usp=sharing) / [log](https://drive.google.com/file/d/14nioV3_VV9KehmxK7XcAHxM8X6JH5WIr/view?usp=sharing) |
| MapTR-tiny | R50 | bevformer |24ep | 48.7 | 18.1 | 10219M (bs 4)|[config](projects/configs/maptr/maptr_tiny_r50_24e_bevformer.py) |[model](https://drive.google.com/file/d/1y-UBwGBSb2xiV40AuQEBhB-xJyV7VusX/view?usp=sharing) / [log](https://drive.google.com/file/d/1r35bRhTGVtyZTP8drXBTOIhLYGCzjEaF/view?usp=sharing) |
## Qualitative results on nuScenes val set
<div align="center"><h4>MapTR maintains stable and robust map construction quality in various driving scenes.</h4></div>

![visualizations](assets/visualizations.png "visualizations")


### *Sunny&Cloudy*
https://user-images.githubusercontent.com/31960625/187059686-11e4dd4b-46db-4411-b680-17ed6deebda2.mp4

### *Rainy*
https://user-images.githubusercontent.com/31960625/187059697-94622ddb-e76a-4fa7-9c44-a688d2e439c0.mp4

### *Night*
https://user-images.githubusercontent.com/31960625/187059706-f7f5a7d8-1d1d-46e0-8be3-c770cf96d694.mp4

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
