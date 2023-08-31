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
import networkx as nx
from av2.map.map_primitives import Polyline
from nuscenes.map_expansion.map_api import NuScenesMapExplorer
from shapely import affinity, ops
from shapely.geometry import Polygon, LineString, box, MultiPolygon, MultiLineString
from shapely.strtree import STRtree
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from av2.geometry.se3 import SE3
import numpy as np
import math
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from scipy.spatial import distance
import warnings
warnings.filterwarnings("ignore")


CAM_NAMES = ['ring_front_center', 'ring_front_right', 'ring_front_left',
    'ring_rear_right','ring_rear_left', 'ring_side_right', 'ring_side_left',
    # 'stereo_front_left', 'stereo_front_right',
    ]
# some fail logs as stated in av2
# https://github.com/argoverse/av2-api/blob/05b7b661b7373adb5115cf13378d344d2ee43906/src/av2/map/README.md#training-online-map-inference-models
FAIL_LOGS = [
    # official
    '75e8adad-50a6-3245-8726-5e612db3d165',
    '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
    'af170aac-8465-3d7b-82c5-64147e94af7d',
    '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
    # observed
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
            '--pc-range',
            type=float,
            nargs='+',
            default=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0],
            help='specify the perception point cloud range')
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
                        num_multithread=64,
                        pc_range = [-30.0, -15.0, -5.0, 30.0, 15.0, 3.0]):
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
    fn = partial(get_data_from_logid, loader=loader, data_root=root_path, pc_range=pc_range)
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

    print('collected in {}s'.format(time.time()-start_time))
    infos = dict(samples=samples)

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

def get_data_from_logid(log_id, 
                        loader: AV2SensorDataLoader, 
                        data_root,
                        pc_range = [-30.0, -15.0, -5.0, 30.0, 15.0, 3.0]):
    samples = []
    discarded = 0
    
    log_map_dirpath = Path(osp.join(data_root, log_id, "map"))
    vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
    if not len(vector_data_fnames) == 1:
        raise RuntimeError(f"JSON file containing vector map data is missing (searched in {log_map_dirpath})")
    vector_data_fname = vector_data_fnames[0]
    vector_data_json_path = vector_data_fname
    avm = ArgoverseStaticMap.from_json(vector_data_json_path)
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
            cam_timestamp_ns = int(cam_ring_fpath[i].stem)
            cam_city_SE3_ego = loader.get_city_SE3_ego(log_id, cam_timestamp_ns)
            cams[cam_name] = dict(
                img_fpath=str(cam_ring_fpath[i]),
                intrinsics=pinhole_cam.intrinsics.K,
                extrinsics=pinhole_cam.extrinsics,
                e2g_translation = cam_city_SE3_ego.translation,
                e2g_rotation = cam_city_SE3_ego.rotation,
            )
        
        city_SE3_ego = loader.get_city_SE3_ego(log_id, int(ts))
        e2g_translation = city_SE3_ego.translation
        e2g_rotation = city_SE3_ego.rotation
        info = dict(
            e2g_translation=e2g_translation,
            e2g_rotation=e2g_rotation,
            cams=cams, 
            lidar_path=str(lidar_fpath),
            # map_fpath=map_fname,
            timestamp=str(ts),
            log_id=log_id,
            token=str(log_id+'_'+str(ts)))

        map_anno = extract_local_map(avm, e2g_translation, e2g_rotation, pc_range)
        info["annotation"] = map_anno

        samples.append(info)

    return samples, discarded

def extract_local_map(avm, e2g_translation, e2g_rotation, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    patch_size = (patch_h, patch_w)
    map_pose = e2g_translation[:2]
    rotation = Quaternion._from_matrix(e2g_rotation)
    patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])
    patch_angle = quaternion_yaw(rotation) / np.pi * 180

    city_SE2_ego = SE3(e2g_rotation, e2g_translation)
    ego_SE3_city = city_SE2_ego.inverse()

    nearby_centerlines = generate_nearby_centerlines(avm, patch_box,patch_angle)
    nearby_dividers = generate_nearby_dividers(avm, patch_box,patch_angle)
    
    map_anno=dict(
        divider=[],
        ped_crossing=[],
        boundary=[],
        centerline=[],
    )
    map_anno['ped_crossing'] = extract_local_ped_crossing(avm, ego_SE3_city, patch_box, patch_angle,patch_size)
    map_anno['boundary'] = extract_local_boundary(avm, ego_SE3_city, patch_box, patch_angle,patch_size)
    map_anno['centerline'] = extract_local_centerline(nearby_centerlines, ego_SE3_city, patch_box, patch_angle,patch_size)
    map_anno['divider'] = extract_local_divider(nearby_dividers, ego_SE3_city, patch_box, patch_angle,patch_size)


    return map_anno

def generate_nearby_centerlines(avm, patch_box, patch_angle):
    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    scene_ls_list = avm.get_scenario_lane_segments()
    scene_ls_dict = dict()
    for ls in scene_ls_list:
        scene_ls_dict[ls.id] = dict(
            ls=ls,
            polygon = Polygon(ls.polygon_boundary),
            predecessors=ls.predecessors,
            successors=ls.successors
        )
    ls_dict = dict()
    for key, value in scene_ls_dict.items():
        polygon = value['polygon']
        if polygon.is_valid:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                ls_dict[key]=value

    for key,value in ls_dict.items():
        value['centerline'] = Polyline.from_array(avm.get_lane_segment_centerline(key).round(3))
    pts_G = nx.DiGraph()
    junction_pts_list = []
    tmp=ls_dict
    for key, value in tmp.items():
        centerline_geom = LineString(value['centerline'].xyz)
        centerline_pts = np.array(centerline_geom.coords).round(3)
        start_pt = centerline_pts[0]
        end_pt = centerline_pts[-1]
        for idx, pts in enumerate(centerline_pts[:-1]):
            pts_G.add_edge(tuple(centerline_pts[idx]),tuple(centerline_pts[idx+1]))
        valid_incoming_num = 0
        for idx, pred in enumerate(value['predecessors']):
            if pred in tmp.keys():
                valid_incoming_num += 1
                pred_geom = LineString(tmp[pred]['centerline'].xyz)
                pred_pt = np.array(pred_geom.coords).round(3)[-1]
                pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
        if valid_incoming_num > 1:
            junction_pts_list.append(tuple(start_pt))
        valid_outgoing_num = 0
        for idx, succ in enumerate(value['successors']):
            if succ in tmp.keys():
                valid_outgoing_num += 1
                succ_geom = LineString(tmp[succ]['centerline'].xyz)
                succ_pt = np.array(succ_geom.coords).round(3)[0]
                pts_G.add_edge(tuple(end_pt), tuple(succ_pt))
        if valid_outgoing_num > 1:
            junction_pts_list.append(tuple(end_pt))
    roots = (v for v, d in pts_G.in_degree() if d == 0)
    leaves = [v for v, d in pts_G.out_degree() if d == 0]
    all_paths = []
    for root in roots:
        paths = nx.all_simple_paths(pts_G, root, leaves)
        all_paths.extend(paths)


    final_centerline_paths = []
    for path in all_paths:
        merged_line = LineString(path)
        merged_line = merged_line.simplify(0.2, preserve_topology=True)
        final_centerline_paths.append(merged_line)

    local_centerline_paths = final_centerline_paths
    return local_centerline_paths

def generate_nearby_dividers(avm, patch_box, patch_angle):
    def get_path(ls_dict):
        pts_G = nx.DiGraph()
        junction_pts_list = []
        tmp=ls_dict
        for key, value in tmp.items():
            centerline_geom = LineString(value['centerline'].xyz)
            centerline_pts = np.array(centerline_geom.coords).round(3)
            start_pt = centerline_pts[0]
            end_pt = centerline_pts[-1]
            for idx, pts in enumerate(centerline_pts[:-1]):
                pts_G.add_edge(tuple(centerline_pts[idx]),tuple(centerline_pts[idx+1]))
            valid_incoming_num = 0
            for idx, pred in enumerate(value['predecessors']):
                if pred in tmp.keys():
                    valid_incoming_num += 1
                    pred_geom = LineString(tmp[pred]['centerline'].xyz)
                    pred_pt = np.array(pred_geom.coords).round(3)[-1]
                    pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
            if valid_incoming_num > 1:
                junction_pts_list.append(tuple(start_pt))
            valid_outgoing_num = 0
            for idx, succ in enumerate(value['successors']):
                if succ in tmp.keys():
                    valid_outgoing_num += 1
                    succ_geom = LineString(tmp[succ]['centerline'].xyz)
                    succ_pt = np.array(succ_geom.coords).round(3)[0]
                    pts_G.add_edge(tuple(end_pt), tuple(succ_pt))
            if valid_outgoing_num > 1:
                junction_pts_list.append(tuple(end_pt))
        roots = (v for v, d in pts_G.in_degree() if d == 0)
        leaves = [v for v, d in pts_G.out_degree() if d == 0]
        all_paths = []
        for root in roots:
            paths = nx.all_simple_paths(pts_G, root, leaves)
            all_paths.extend(paths)


        final_centerline_paths = []
        for path in all_paths:
            merged_line = LineString(path)
            merged_line = merged_line.simplify(0.2, preserve_topology=True)
            final_centerline_paths.append(merged_line)

        local_centerline_paths = final_centerline_paths
        return local_centerline_paths

    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    scene_ls_list = avm.get_scenario_lane_segments()
    scene_ls_dict = dict()
    for ls in scene_ls_list:
        scene_ls_dict[ls.id] = dict(
            ls=ls,
            polygon = Polygon(ls.polygon_boundary),
            predecessors=ls.predecessors,
            successors=ls.successors
        )
#     nearby_ls_ids = []
    nearby_ls_dict = dict()
    for key, value in scene_ls_dict.items():
        polygon = value['polygon']
        if polygon.is_valid:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                nearby_ls_dict[key] = value['ls']

    ls_dict = nearby_ls_dict
    divider_ls_dict = dict()
    for key, value in ls_dict.items():
        if not value.is_intersection:
            divider_ls_dict[key] = value

    left_lane_dict = {}
    right_lane_dict = {}
    for key,value in divider_ls_dict.items():
        if value.left_neighbor_id is not None:
            left_lane_dict[key] = dict(
                polyline=value.left_lane_boundary,
                predecessors = value.predecessors,
                successors = value.successors,
                left_neighbor_id = value.left_neighbor_id,
            )
        if value.right_neighbor_id is not None:
            right_lane_dict[key] = dict(
                polyline = value.right_lane_boundary,
                predecessors = value.predecessors,
                successors = value.successors,
                right_neighbor_id = value.right_neighbor_id,
            )
    for key, value in left_lane_dict.items():
        if value['left_neighbor_id'] in right_lane_dict.keys():
            del right_lane_dict[value['left_neighbor_id']]

    for key, value in right_lane_dict.items():
        if value['right_neighbor_id'] in left_lane_dict.keys():
            del left_lane_dict[value['right_neighbor_id']]
    
    for key, value in left_lane_dict.items():
        value['centerline'] = value['polyline']

    for key, value in right_lane_dict.items():
        value['centerline'] = value['polyline']
    
    left_paths = get_path(left_lane_dict)
    right_paths = get_path(right_lane_dict)
    local_dividers = left_paths + right_paths

    return local_dividers

def proc_polygon(polygon, ego_SE3_city):
    # import pdb;pdb.set_trace()
    interiors = []
    exterior_cityframe = np.array(list(polygon.exterior.coords))
    exterior_egoframe = ego_SE3_city.transform_point_cloud(exterior_cityframe)
    for inter in polygon.interiors:
        inter_cityframe = np.array(list(inter.coords))
        inter_egoframe = ego_SE3_city.transform_point_cloud(inter_cityframe)
        interiors.append(inter_egoframe[:,:3])

    new_polygon = Polygon(exterior_egoframe[:,:3], interiors)
    return new_polygon
def proc_line(line,ego_SE3_city):
    # import pdb;pdb.set_trace()
    new_line_pts_cityframe = np.array(list(line.coords))
    new_line_pts_egoframe = ego_SE3_city.transform_point_cloud(new_line_pts_cityframe)
    line = LineString(new_line_pts_egoframe[:,:3]) #TODO
    return line

def extract_local_centerline(nearby_centerlines, ego_SE3_city, patch_box, patch_angle,patch_size):
      
    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    line_list = []
    for line in nearby_centerlines:
        if line.is_empty:  # Skip lines without nodes.
            continue
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            if new_line.geom_type == 'MultiLineString':
                for single_line in new_line.geoms:
                    if single_line.is_empty:
                        continue
                    single_line = proc_line(single_line,ego_SE3_city)
                    line_list.append(single_line)
            else:
                new_line = proc_line(new_line, ego_SE3_city)
                line_list.append(new_line)
                
    centerlines = line_list
    
    poly_centerlines = [line.buffer(1,
                cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre) for line in centerlines]
    index_by_id = dict((id(pt), i) for i, pt in enumerate(poly_centerlines))
    tree = STRtree(poly_centerlines)
    final_pgeom = []
    remain_idx = [i for i in range(len(centerlines))]
    for i, pline in enumerate(poly_centerlines):
        if i not in remain_idx:
            continue
        remain_idx.pop(remain_idx.index(i))

        final_pgeom.append(centerlines[i])
        for o in tree.query(pline):
            o_idx = index_by_id[id(o)]
            if o_idx not in remain_idx:
                continue
            inter = o.intersection(pline).area
            union = o.union(pline).area
            iou = inter / union
            if iou >= 0.90:
                remain_idx.pop(remain_idx.index(o_idx))
    return [np.array(line.coords) for line in final_pgeom]

def merge_dividers(divider_list):
    # divider_list: List[np.array(N,3)]
    if len(divider_list) < 2:
        return divider_list
    divider_list_shapely = [LineString(divider) for divider in divider_list]
    poly_dividers = [divider.buffer(1,
                cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre) for divider in divider_list_shapely]
    tree = STRtree(poly_dividers)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(poly_dividers))
    final_pgeom = []
    remain_idx = [i for i in range(len(poly_dividers))]
    for i, pline in enumerate(poly_dividers):
        if i not in remain_idx:
            continue
        remain_idx.pop(remain_idx.index(i))
        final_pgeom.append(divider_list[i])
        for o in tree.query(pline):
            o_idx = index_by_id[id(o)]
            if o_idx not in remain_idx:
                continue
            # remove highly overlap divider
            inter = o.intersection(pline).area
            o_iof = inter / o.area
            p_iof = inter / pline.area
            # if query divider is highly overlaped with latter dividers, just remove it
            if p_iof >=0.95:
                final_pgeom.pop()
                break
            # if queried divider is highly overlapped with query divider,
            # drop it and just turn to next one.
            if o_iof >= 0.95:
                remain_idx.pop(remain_idx.index(o_idx))
                continue

            pline_se_pts = final_pgeom[-1][[0,-1],:2] # only on xy
            o_se_pts = divider_list[o_idx][[0,-1],:2] # only on xy
            four_se_pts = np.concatenate([pline_se_pts,o_se_pts],axis=0)
            dist_mat = distance.cdist(four_se_pts, four_se_pts, 'euclidean')
            for j in range(4):
                dist_mat[j,j] = 100
            index = np.where(dist_mat==0)[0].tolist()
            if index == [0, 2]:
                # e oline s s pline e
                # +-------+ +-------+
                final_pgeom[-1] = np.concatenate([np.flip(divider_list[o_idx], axis=0)[:-1], final_pgeom[-1]])
                remain_idx.pop(remain_idx.index(o_idx))
            elif index == [1, 2]:
                # s pline e s oline e
                # +-------+ +-------+
                final_pgeom[-1] = np.concatenate([final_pgeom[-1][:-1], divider_list[o_idx]])
                remain_idx.pop(remain_idx.index(o_idx))
            elif index == [0, 3]:
                # s oline e s pline e
                # +-------+ +-------+
                final_pgeom[-1] = np.concatenate([divider_list[o_idx][:-1], final_pgeom[-1]])
                remain_idx.pop(remain_idx.index(o_idx))
            elif index == [1, 3]:
                # s pline e e oline s
                # +-------+ +-------+
                final_pgeom[-1]  = np.concatenate([final_pgeom[-1][:-1],np.flip(divider_list[o_idx], axis=0)])
                remain_idx.pop(remain_idx.index(o_idx))
            elif len(index) > 2:
                remain_idx.pop(remain_idx.index(o_idx))

    return final_pgeom


def extract_local_divider(nearby_dividers, ego_SE3_city, patch_box, patch_angle,patch_size):
    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    line_list = []
    for line in nearby_dividers:
        if line.is_empty:  # Skip lines without nodes.
            continue
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            if new_line.geom_type == 'MultiLineString':
                for single_line in new_line.geoms:
                    if single_line.is_empty:
                        continue
                    single_line = proc_line(single_line,ego_SE3_city)
                    line_list.append(single_line)
            else:
                new_line = proc_line(new_line, ego_SE3_city)
                line_list.append(new_line)
                
    centerlines = line_list
    
    poly_centerlines = [line.buffer(1,
                cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre) for line in centerlines]
    index_by_id = dict((id(pt), i) for i, pt in enumerate(poly_centerlines))
    tree = STRtree(poly_centerlines)
    final_pgeom = []
    remain_idx = [i for i in range(len(centerlines))]
    for i, pline in enumerate(poly_centerlines):
        if i not in remain_idx:
            continue
        remain_idx.pop(remain_idx.index(i))

        final_pgeom.append(centerlines[i])
        for o in tree.query(pline):
            o_idx = index_by_id[id(o)]
            if o_idx not in remain_idx:
                continue
            inter = o.intersection(pline).area
            union = o.union(pline).area
            iou = inter / union
            if iou >= 0.90:
                remain_idx.pop(remain_idx.index(o_idx))
    return [np.array(line.coords) for line in final_pgeom]
def extract_local_boundary(avm, ego_SE3_city, patch_box, patch_angle,patch_size):
    boundary_list = []
    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    for da in avm.get_scenario_vector_drivable_areas():
        boundary_list.append(da.xyz)

    polygon_list = []
    for da in boundary_list:
        exterior_coords = da
        interiors = []
    #     polygon = Polygon(exterior_coords, interiors)
        polygon = Polygon(exterior_coords, interiors)
        if polygon.is_valid:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                if new_polygon.geom_type is 'Polygon':
                    if not new_polygon.is_valid:
                        continue
                    new_polygon = proc_polygon(new_polygon,ego_SE3_city)
                    if not new_polygon.is_valid:
                        continue
                elif new_polygon.geom_type is 'MultiPolygon':
                    polygons = []
                    for single_polygon in new_polygon.geoms:
                        if not single_polygon.is_valid or single_polygon.is_empty:
                            continue
                        new_single_polygon = proc_polygon(single_polygon,ego_SE3_city)
                        if not new_single_polygon.is_valid:
                            continue
                        polygons.append(new_single_polygon)
                    if len(polygons) == 0:
                        continue
                    new_polygon = MultiPolygon(polygons)
                    if not new_polygon.is_valid:
                        continue
                else:
                    raise ValueError('{} is not valid'.format(new_polygon.geom_type))

                if new_polygon.geom_type is 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                polygon_list.append(new_polygon)

    union_segments = ops.unary_union(polygon_list)
    max_x = patch_size[1] / 2
    max_y = patch_size[0] / 2
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    exteriors = []
    interiors = []
    if union_segments.geom_type != 'MultiPolygon':
        union_segments = MultiPolygon([union_segments])
    for poly in union_segments.geoms:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)


    results = []
    for ext in exteriors:
        if ext.is_ccw:
            ext.coords = list(ext.coords)[::-1]
        lines = ext.intersection(local_patch)
        if isinstance(lines, MultiLineString):
            lines = ops.linemerge(lines)
        results.append(lines)

    for inter in interiors:
        if not inter.is_ccw:
            inter.coords = list(inter.coords)[::-1]
        lines = inter.intersection(local_patch)
        if isinstance(lines, MultiLineString):
            lines = ops.linemerge(lines)
        results.append(lines)

    boundary_lines = []
    for line in results:
        if not line.is_empty:
            if line.geom_type == 'MultiLineString':
                for single_line in line.geoms:
                    boundary_lines.append(np.array(single_line.coords))
            elif line.geom_type == 'LineString':
                boundary_lines.append(np.array(line.coords))
            else:
                raise NotImplementedError
    return boundary_lines



def extract_local_ped_crossing(avm, ego_SE3_city, patch_box, patch_angle,patch_size):
    ped_list = []
    for pc in avm.get_scenario_ped_crossings():
        ped_list.append(pc.polygon)

    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)

    polygon_list = []
    for pc in ped_list:
        exterior_coords = pc
        interiors = []
        polygon = Polygon(exterior_coords, interiors)
        if polygon.is_valid:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                if new_polygon.geom_type is 'Polygon':
                    if not new_polygon.is_valid:
                        continue
                    new_polygon = proc_polygon(new_polygon,ego_SE3_city)
                    if not new_polygon.is_valid:
                        continue
                elif new_polygon.geom_type is 'MultiPolygon':
                    polygons = []
                    for single_polygon in new_polygon.geoms:
                        if not single_polygon.is_valid or single_polygon.is_empty:
                            continue
                        new_single_polygon = proc_polygon(single_polygon,ego_SE3_city)
                        if not new_single_polygon.is_valid:
                            continue
                        polygons.append(new_single_polygon)
                    if len(polygons) == 0:
                        continue
                    new_polygon = MultiPolygon(polygons)
                    if not new_polygon.is_valid:
                        continue
                else:
                    raise ValueError('{} is not valid'.format(new_polygon.geom_type))

                if new_polygon.geom_type is 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                polygon_list.append(new_polygon)


    def get_rec_direction(geom):
        rect = geom.minimum_rotated_rectangle # polygon as rotated rect
        rect_v_p = np.array(rect.exterior.coords)[:3] # vector point
        rect_v = rect_v_p[1:]-rect_v_p[:-1] # vector 
        v_len = np.linalg.norm(rect_v, axis=-1) # vector length
        longest_v_i = v_len.argmax()

        return rect_v[longest_v_i], v_len[longest_v_i]

    ped_geoms = polygon_list
    tree = STRtree(ped_geoms)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))
    final_pgeom = []
    remain_idx = [i for i in range(len(ped_geoms))]
    for i, pgeom in enumerate(ped_geoms):
        if i not in remain_idx:
            continue
        remain_idx.pop(remain_idx.index(i))
        pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
        final_pgeom.append(pgeom)
        for o in tree.query(pgeom):
            o_idx = index_by_id[id(o)]
            if o_idx not in remain_idx:
                continue
            o_v, o_v_norm = get_rec_direction(o)
            cos = pgeom_v.dot(o_v)/(pgeom_v_norm*o_v_norm)
            if 1 - np.abs(cos) < 0.01:  # theta < 8 degrees.
                final_pgeom[-1] =\
                    final_pgeom[-1].union(o) # union parallel ped?
                # update
                remain_idx.pop(remain_idx.index(o_idx))
    for i in range(len(final_pgeom)):
        if final_pgeom[i].geom_type != 'MultiPolygon':
            final_pgeom[i] = MultiPolygon([final_pgeom[i]])

    max_x = patch_size[1] / 2
    max_y = patch_size[0] / 2
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    # results = []
    results = []
    for geom in final_pgeom:
        for ped_poly in geom.geoms:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)

            if lines.type != 'LineString':
                lines = ops.linemerge(lines)

            # same instance but not connected.
            if lines.type != 'LineString':
                ls = []
                for l in lines.geoms:
                    ls.append(np.array(l.coords))

                lines = np.concatenate(ls, axis=0)
                lines = LineString(lines)

            results.append(np.array(lines.coords))
    return results
if __name__ == '__main__':
    args = parse_args()
    for name in ['train', 'val', 'test']:
        create_av2_infos_mp(
            root_path=args.data_root,
            split=name,
            info_prefix='av2',
            dest_path=args.data_root,
            pc_range=args.pc_range,)