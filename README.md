<div align="center">
<h1>MapTR <img src="assets/map.png" width="30"></h1>
<h3>Structured Modeling and Learning for Online Vectorized HD Map Construction</h3>

[Bencheng Liao](https://github.com/LegendBC)<sup>1,2,3</sup> \*, [Shaoyu Chen](https://scholar.google.com/citations?user=PIeNN2gAAAAJ&hl=en&oi=sra)<sup>1,3</sup> \*, [Xinggang Wang](https://xinggangw.info/)<sup>1 :email:</sup>, [Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ&hl=zh-CN)<sup>1,3</sup>, [Qian Zhang](https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN)<sup>3</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, [Chang Huang](https://scholar.google.com/citations?user=IyyEKyIAAAAJ&hl=zh-CN)<sup>3</sup>
 
<sup>1</sup> School of EIC, HUST, <sup>2</sup> Institute of Artificial Intelligence, HUST, <sup>3</sup> Horizon Robotics

(\*) equal contribution, (<sup>:email:</sup>) corresponding author.

ArXiv Preprint ([arXiv XXXX.XXXX](https://arxiv.org/abs/XXXX.XXXXX))

</div>

#
### News

* **`X Aug., 2022`:** We released our paper on Arxiv. Code/Models are coming soon. Please stay tuned! ☕️


## Introduction
<div align="center"><h4>MapTR is a simple, fast and strong online vectorized HD map construction framework.</h4></div>

![framework](assets/framework.png "framework")
We present MapTR, a structured end-to-end framework for efficient online vectorized HD map construction. We propose a unified permutation-based modeling approach, *i.e.*, modeling map element as a point set with a group of equivalent permutations, which avoids the definition ambiguity and eases learning. We adopt a
hierarchical query embedding scheme to flexibly encode structured map information and perform hierarchical bipartite matching for map element learning. MapTR achieves new state-of-the-art vectorized map construction performance and keeps real-time inference speed. 
Even with only camera input, MapTR-tiny significantly outperforms multi-modality counterparts by $13.5$ AP. MapTR-nano achieves SOTA camera-based performance ( $44.2$ mAP) and runs at  $25.1$ FPS. To our best knowledge, MapTR is the first approach realizing real-time vectorized HD map construction. 
## Models
| Method | Backbone | Lr Schd | mAP| FPS|memroy | Config | Download |
| :---: | :---: | :---: | :---: | :---:|:---:| :---: | :---: |
| MapTR-nano | R18 | 110ep | 44.2 | 25.1| 11907M (bs 24) |[coming soon] |[coming soon] |
| MapTR-tiny | R50 | 24ep | 50.3 | 11.2| 10287M (bs 4) | [coming soon]|[coming soon] |
| MapTR-tiny | R50 | 110ep | 58.7|11.2| 10287M (bs 4)|[coming soon] |[coming soon] |

**Notes**: 

- FPS is measured on NVIDIA GTX3090 GPU
- All the experiments are performed on 8 NVIDIA GTX3090 GPUs

## Qualitative results on nuScenes val set
<div align="center"><h4>MapTR maintains stable and robust map construction quality in various driving scenes.</h4></div>

![visualizations](assets/visualizations.png "visualizations")


### *Sunny&Cloudy*
https://user-images.githubusercontent.com/31960625/187059686-11e4dd4b-46db-4411-b680-17ed6deebda2.mp4

### *Rainy*
https://user-images.githubusercontent.com/31960625/187059697-94622ddb-e76a-4fa7-9c44-a688d2e439c0.mp4

### *Night*
https://user-images.githubusercontent.com/31960625/187059706-f7f5a7d8-1d1d-46e0-8be3-c770cf96d694.mp4


## Usage
coming soon

## Citation
If you find MapTR is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```bibtex
@article{MapTR,
  title={MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction},
  author={Liao, Bencheng and Chen, shaoyu and Wang, Xinggang and Cheng, Tianheng, and Zhang, Qian and Liu, Wenyu and Huang, Chang},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2022}
}
```
