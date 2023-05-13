from functools import partial
from multiprocessing import Pool
import multiprocessing
from random import sample
import time
import mmcv
import logging
from pathlib import Path
from os import path as osp
import os
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.lane_segment import LaneMarkType, LaneSegment
from av2.map.map_api import ArgoverseStaticMap
from tqdm import tqdm
import argparse

CAM_NAMES = ['ring_front_center', 'ring_front_right', 'ring_front_left',
    'ring_rear_right','ring_rear_left', 'ring_side_right', 'ring_side_left',
    # 'stereo_front_left', 'stereo_front_right',
    ]
# some fail logs as stated in av2
# https://github.com/argoverse/av2-api/blob/05b7b661b7373adb5115cf13378d344d2ee43906/src/av2/map/README.md#training-online-map-inference-models
FAIL_LOGS = [
    '75e8adad-50a6-3245-8726-5e612db3d165',
    '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
    'af170aac-8465-3d7b-82c5-64147e94af7d',
    '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
    '01bb304d-7bd8-35f8-bbef-7086b688e35e',
    '453e5558-6363-38e3-bf9b-42b5ba0a6f1d'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        help='specify the root path of dataset')
    parser.add_argument(
        '--nproc',
        type=int,
        default=64,
        required=False,
        help='workers to process data')
    args = parser.parse_args()
    return args

def create_av2_infos_mp(root_path,
                        info_prefix,
                        dest_path=None,
                        split='train',
                        num_multithread=64):
    """Create info file of av2 dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        dest_path (str): Path to store generated file, default to root_path
        split (str): Split of the data.
            Default: 'train'
    """
    root_path = osp.join(root_path, split)
    if dest_path is None:
        dest_path = root_path
    
    loader = AV2SensorDataLoader(Path(root_path), Path(root_path))
    log_ids = list(loader.get_log_ids())
    # import pdb;pdb.set_trace()
    for l in FAIL_LOGS:
        if l in log_ids:
            log_ids.remove(l)

    print('collecting samples...')
    start_time = time.time()
    print('num cpu:', multiprocessing.cpu_count())
    print(f'using {num_multithread} threads')

    # to supress logging from av2.utils.synchronization_database
    sdb_logger = logging.getLogger('av2.utils.synchronization_database')
    prev_level = sdb_logger.level
    sdb_logger.setLevel(logging.CRITICAL)

    # FIXME: need to check the order
    pool = Pool(num_multithread)
    fn = partial(get_data_from_logid, loader=loader, data_root=root_path)
    rt = pool.map_async(fn, log_ids)
    pool.close()
    pool.join()
    results = rt.get()

    samples = []
    discarded = 0
    sample_idx = 0
    for _samples, _discarded in results:
        for i in range(len(_samples)):
            _samples[i]['sample_idx'] = sample_idx
            sample_idx += 1
        samples += _samples
        discarded += _discarded
    
    sdb_logger.setLevel(prev_level)
    print(f'{len(samples)} available samples, {discarded} samples discarded')

    id2map = {}
    for log_id in log_ids:
        log_map_dirpath = Path(osp.join(root_path, log_id, "map"))
        vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
        # vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
        if not len(vector_data_fnames) == 1:
            raise RuntimeError(f"JSON file containing vector map data is missing (searched in {log_map_dirpath})")
        vector_data_fname = vector_data_fnames[0]
        vector_data_json_path = vector_data_fname
        avm = ArgoverseStaticMap.from_json(vector_data_json_path)
        # import pdb;pdb.set_trace()
        map_elements = {}
        map_elements['divider'] = get_divider(avm)
        map_elements['ped_crossing'] = get_ped(avm)
        map_elements['boundary'] = get_boundary(avm)

        # map_fname = osp.join(map_path_dir, map_fname)
        id2map[log_id] = map_elements

    print('collected in {}s'.format(time.time()-start_time))
    infos = dict(samples=samples, id2map=id2map)

    info_path = osp.join(dest_path,
                                 '{}_map_infos_{}.pkl'.format(info_prefix, split))
    print(f'saving results to {info_path}')
    mmcv.dump(infos, info_path)
    # mmcv.dump(samples, info_path)

def get_divider(avm):
    divider_list = []
    for ls in avm.get_scenario_lane_segments():
            for bound_type, bound_city in zip([ls.left_mark_type, ls.right_mark_type], [ls.left_lane_boundary, ls.right_lane_boundary]):
                if bound_type not in [LaneMarkType.NONE,]:
                    divider_list.append(bound_city.xyz)
    return divider_list

def get_boundary(avm):
    boundary_list = []
    for da in avm.get_scenario_vector_drivable_areas():
        boundary_list.append(da.xyz)
    return boundary_list

def get_ped(avm):
    ped_list = []
    for pc in avm.get_scenario_ped_crossings():
        ped_list.append(pc.polygon)
    return ped_list

def get_data_from_logid(log_id, loader: AV2SensorDataLoader, data_root):
    samples = []
    discarded = 0
    
    # We use lidar timestamps to query all sensors.
    # The frequency is 10Hz
    cam_timestamps = loader._sdb.per_log_lidar_timestamps_index[log_id]
    for ts in cam_timestamps:
        cam_ring_fpath = [loader.get_closest_img_fpath(
                log_id, cam_name, ts
            ) for cam_name in CAM_NAMES]
        lidar_fpath = loader.get_closest_lidar_fpath(log_id, ts)

        # If bad sensor synchronization, discard the sample
        if None in cam_ring_fpath or lidar_fpath is None:
            discarded += 1
            continue

        cams = {}
        for i, cam_name in enumerate(CAM_NAMES):
            pinhole_cam = loader.get_log_pinhole_camera(log_id, cam_name)
            cams[cam_name] = dict(
                img_fpath=str(cam_ring_fpath[i]),
                intrinsics=pinhole_cam.intrinsics.K,
                extrinsics=pinhole_cam.extrinsics,
            )
        
        city_SE3_ego = loader.get_city_SE3_ego(log_id, int(ts))
        e2g_translation = city_SE3_ego.translation
        e2g_rotation = city_SE3_ego.rotation
        
        samples.append(dict(
            e2g_translation=e2g_translation,
            e2g_rotation=e2g_rotation,
            cams=cams, 
            lidar_fpath=str(lidar_fpath),
            # map_fpath=map_fname,
            timestamp=str(ts),
            log_id=log_id,
            token=str(log_id+'_'+str(ts))))

    return samples, discarded


if __name__ == '__main__':
    args = parse_args()
    for name in ['train', 'val', 'test']:
        create_av2_infos_mp(
            root_path=args.data_root,
            split=name,
            info_prefix='av2',
            dest_path=args.data_root,)