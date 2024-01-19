# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: September 05, 2023
#
# This code is licensed under the MIT License.
# ==============================================================================

import numpy as np
from plyfile import PlyData, PlyElement
import argparse

def load_npz_file(npz_filename):
    data = np.load(npz_filename)
    point_cloud = data['coords']
    colors = np.stack((data['R'], data['G'], data['B']), axis=-1)
    colors = (colors * 255).astype(np.uint8)
    return point_cloud, colors

def save_point_cloud_as_ply(point_cloud, colors, ply_filename):
    vertex_dtype = np.dtype([
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), 
    ])
    vertex_array = np.array(
        list(zip(
            point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
            # colors[:, 0], colors[:, 1], colors[:, 2], colors[:, 3],
            colors[:, 0], colors[:, 1], colors[:, 2],
            # material_alpha
        )),
        dtype=vertex_dtype
    )

    ply_element = PlyElement.describe(vertex_array, 'vertex')
    PlyData([ply_element], text=True).write(ply_filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_filename', type = str, required = True)
    parser.add_argument('--ply_filename', type = str, required = True)
    args = parser.parse_args()

    point_cloud, colors = load_npz_file(args.npz_filename)
    save_point_cloud_as_ply(point_cloud, colors, args.ply_filename)

if __name__ == '__main__':
    main()