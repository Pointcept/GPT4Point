import tempfile
from contextlib import contextmanager
from typing import Iterator, Optional, Union

import blobfile as bf
import numpy as np
import torch
from PIL import Image

from shap_e.rendering.blender.render import render_mesh, render_model
from shap_e.rendering.blender.view_data import BlenderViewData
from shap_e.rendering.mesh import TriMesh
from shap_e.rendering.point_cloud import PointCloud
from shap_e.rendering.view_data import ViewData
from shap_e.util.collections import AttrDict
from shap_e.util.image_util import center_crop, get_alpha, remove_alpha, resize
from IPython import embed
import time


def load_or_create_multimodal_batch(
    device: torch.device,
    *,
    mesh_path: Optional[str] = None,
    model_path: Optional[str] = None, 
    cache_dir: Optional[str] = None,
    point_count: int = 2**14,
    random_sample_count: int = 2**19,
    pc_num_views: int = 40,
    mv_light_mode: Optional[str] = None,
    mv_num_views: int = 20,
    mv_image_size: int = 512,
    mv_alpha_removal: str = "black",
    verbose: bool = False,
) -> AttrDict:
    if verbose:
        print("creating point cloud...")
    pc = load_or_create_pc(
        mesh_path=mesh_path,
        model_path=model_path,
        cache_dir=cache_dir,
        random_sample_count=random_sample_count,
        point_count=point_count,
        num_views=pc_num_views,
        verbose=verbose,
    )
    return pc

def load_or_create_pc(
    *,
    mesh_path: Optional[str],
    model_path: Optional[str],
    cache_dir: Optional[str],
    random_sample_count: int,
    point_count: int,
    num_views: int,
    verbose: bool = False,
) -> PointCloud:

    assert (model_path is not None) ^ (
        mesh_path is not None
    ), "must specify exactly one of model_path or mesh_path"
    path = model_path if model_path is not None else mesh_path

    if cache_dir is not None:
        cache_path = bf.join(
            cache_dir,
            f"pc_{bf.basename(path)}_mat_{num_views}_{random_sample_count}_{point_count}.npz",
        )
        if bf.exists(cache_path):
            return PointCloud.load(cache_path)
    else:
        cache_path = None

    with load_or_create_multiview(
        mesh_path=mesh_path,
        model_path=model_path,
        cache_dir=cache_dir,
        num_views=num_views,
        verbose=verbose,
    ) as mv:
        if verbose:
            print("extracting point cloud from multiview...")
        pc = mv_to_pc(
            multiview=mv, random_sample_count=random_sample_count, point_count=point_count
        )
        if cache_path is not None and pc is not None:
            pc.save(cache_path)
        return pc


@contextmanager
def load_or_create_multiview(
    *,
    mesh_path: Optional[str],
    model_path: Optional[str],
    cache_dir: Optional[str],
    num_views: int = 20,
    extract_material: bool = True,
    light_mode: Optional[str] = None,
    verbose: bool = False,
) -> Iterator[BlenderViewData]:

    assert (model_path is not None) ^ (
        mesh_path is not None
    ), "must specify exactly one of model_path or mesh_path"
    path = model_path if model_path is not None else mesh_path

    if extract_material:
        assert light_mode is None, "light_mode is ignored when extract_material=True"
    else:
        assert light_mode is not None, "must specify light_mode when extract_material=False"

    if cache_dir is not None:
        cache_path = model_path
        if bf.exists(cache_path):
            with bf.BlobFile(cache_path, "rb") as f:
                uid = cache_path.split('/')[-1].split('.')[0]
                yield BlenderViewData(f, uid)
                return
    else:
        cache_path = None

def mv_to_pc(multiview: ViewData, random_sample_count: int, point_count: int) -> PointCloud:
    pc = PointCloud.from_rgbd(multiview)

    if pc == None:
        return None
    # Handle empty samples.
    if len(pc.coords) == 0:
        pc = PointCloud(
            coords=np.zeros([1, 3]),
            channels=dict(zip("RGB", np.zeros([3, 1]))),
        )
    while len(pc.coords) < point_count:
        pc = pc.combine(pc)
        # Prevent duplicate points; some models may not like it.
        pc.coords += np.random.normal(size=pc.coords.shape) * 1e-4

    #embed()
    #exit()

    s = time.time()
    # pc = pc.random_sample(random_sample_count)
    pc = pc.random_sample(int(random_sample_count/4))
    # from pointnet2_ops.pointnet2_utils import FurthestPointSampling
    # pc = pc.farthest_point_sample(int(point_count/4), average_neighbors=False)
    pc = pc.farthest_point_sample(int(point_count), average_neighbors=False)
    # pc = pc.farthest_point_sample(point_count, average_neighbors=True)
    print('farthest_point_sample', time.time() - s)

    return pc


def normalize_input_batch(batch: AttrDict, *, pc_scale: float, color_scale: float) -> AttrDict:
    res = batch.copy()
    scale_vec = torch.tensor([*([pc_scale] * 3), *([color_scale] * 3)], device=batch.points.device)
    res.points = res.points * scale_vec[:, None]

    if "cameras" in res:
        res.cameras = [[cam.scale_scene(pc_scale) for cam in cams] for cams in res.cameras]

    if "depths" in res:
        res.depths = [[depth * pc_scale for depth in depths] for depths in res.depths]

    return res


def process_depth(depth_img: np.ndarray, image_size: int) -> np.ndarray:
    depth_img = center_crop(depth_img)
    depth_img = resize(depth_img, width=image_size, height=image_size)
    return np.squeeze(depth_img)


def process_image(
    img_or_img_arr: Union[Image.Image, np.ndarray], alpha_removal: str, image_size: int
):
    if isinstance(img_or_img_arr, np.ndarray):
        img = Image.fromarray(img_or_img_arr)
        img_arr = img_or_img_arr
    else:
        img = img_or_img_arr
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            # Grayscale
            rgb = Image.new("RGB", img.size)
            rgb.paste(img)
            img = rgb
            img_arr = np.array(img)

    img = center_crop(img)
    alpha = get_alpha(img)
    img = remove_alpha(img, mode=alpha_removal)
    alpha = alpha.resize((image_size,) * 2, resample=Image.BILINEAR)
    img = img.resize((image_size,) * 2, resample=Image.BILINEAR)
    return img, alpha
