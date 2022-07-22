import os
import json
import argparse
import numpy as np

correct_pose = [-1,1,-1]

def matrix_nerf2ngp(matrix):
    matrix[:, 0] *= correct_pose[0]
    matrix[:, 1] *= correct_pose[1]
    matrix[:, 2] *= correct_pose[2]
    return matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nerf json to ngp")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    file_path = args.config_file + ".json"
    print(f"load {file_path}")
    with open(file_path,'r')as f:
        data=json.load(f)
    for frame in data['frames']:
        matrix=np.array(frame['transform_matrix'], np.float32)
        frame['transform_matrix'] = matrix_nerf2ngp(matrix).tolist()
    out_path = args.config_file + "_ngp.json"
    print(f"save {out_path}")
    with open(out_path, "w") as outfile:
        json.dump(data, outfile, indent=4)
