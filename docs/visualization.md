# Visualization

We provide all the visualization scripts under `/path/to/MapTR/tools/maptr/`

## Visualize prediction

```shell
cd /path/to/MapTR/
export PYTHONPATH="/path/to/MapTR/"
# visualize nuscenes dataset
python tools/maptrv2/nusc_vis_pred.py /path/to/experiment/config /path/to/experiment/ckpt

#visualize argoverse2 dataset
python tools/maptrv2/av2_vis_pred.py /path/to/experiment/config /path/to/experiment/ckpt
```
**Notes**: 

- All the visualization samples will be saved in `/path/to/MapTR/work_dirs/experiment/vis_pred/` automatically. If you want to customize the saving path, you can add `--show-dir /customized_path`.
- The score threshold is set to 0.3 by default. For better visualization, you can adjust the threshold by adding `--score-thresh customized_thresh`
- The GT is visualized in fixed_num_pts format by default, we provide multiple formats to visualize GT at the same time by setting `--gt-format`: `se_pts` means the start and end points of GT, `bbox` means the bounding box envelops the GT, `polyline_pts` means the original annotated GT (you can use Douglas-Peucker algorithm to simplify the redundant annotated points).

## Merge them into video

We also provide the script to merge the input, output and GT into video to benchmark the performance qualitatively.

```shell
python tools/maptr/generate_video.py /path/to/visualization/directory
```
**Notes**: 
- The video will be saved in `/path/to/MapTR/work_dirs/experiment/`