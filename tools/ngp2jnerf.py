import os
import json
import argparse
import numpy as np

scenes = ["Car", "Easyship"]
backs = ["B_test"]
correct_pose = [-1,1,-1]

def matrix_nerf2ngp(matrix):
    matrix[:, 0] *= correct_pose[0]
    matrix[:, 1] *= correct_pose[1]
    matrix[:, 2] *= correct_pose[2]
    return matrix

for scene in scenes:
    for back in backs:
        file_path_in = f'../data/Jrender_Dataset/{scene}/transforms_{back}_ngp.json'
        file_path_out = f'../data/Jrender_Dataset/{scene}/transforms_{back}.json'

        print(f"load {file_path_in}")
        with open(file_path_in,'r')as f:
            data=json.load(f)
        for frame in data['frames']:
            matrix=np.array(frame['transform_matrix'], np.float32)
            frame['transform_matrix'] = matrix_nerf2ngp(matrix).tolist()
        print(f"save {file_path_out}")
        with open(file_path_out, "w") as outfile:
            json.dump(data, outfile, indent=4)
