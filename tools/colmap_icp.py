import os
import json
import argparse
import numpy as np
import open3d as o3d
import quaternion as quat

correct_pose = [-1,1,-1]

def matrix_nerf2ngp(matrix):
    matrix[:, 0] *= correct_pose[0]
    matrix[:, 1] *= correct_pose[1]
    matrix[:, 2] *= correct_pose[2]
    return matrix

def icpIter(data, gt, rot, trans, threshold:float):
    query_c, pair_c = np.zeros((3, 1), dtype = float), np.zeros((3, 1), dtype = float)
    qpt = np.zeros((3, 3), dtype = float)
    query_start_p, base_start_p = rot @ data[0] + trans, gt[0]
    valid_cnt = 0
    max_dist = 0
    max_dist_valid = 0
    avg_scale1, avg_scale2 = 0, 0
    for query_pt, value_pt in zip(data, gt):
        query_pt = rot @ query_pt + trans
        dist = np.linalg.norm(query_pt - value_pt)
        max_dist = max_dist if dist < max_dist else dist
        if dist > threshold:
            continue
        max_dist_valid = max_dist_valid if dist < max_dist_valid else dist
        valid_cnt += 1
        avg_scale1 += np.linalg.norm(query_pt - query_start_p)
        avg_scale2 += np.linalg.norm(value_pt - base_start_p)
        query_c += query_pt
        pair_c += value_pt
        qpt += (query_pt - query_start_p) @ (value_pt - base_start_p).T
    avg_scale = avg_scale2 / avg_scale1
    print(valid_cnt, ' ', max_dist, ' ', max_dist_valid, ' ', avg_scale)
    query_c /= valid_cnt
    pair_c /= valid_cnt
    qpt -= valid_cnt * (query_c - query_start_p) @ (pair_c - base_start_p).T
    u, s, vh = np.linalg.svd(qpt)
    r = vh.T @ u.T
    rr = vh.T @ np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., np.linalg.det(r)]]) @ u.T
    return avg_scale, rr, pair_c - rr @ query_c, max_dist

def icp(data, gt):
    threshold = 20
    scale = 1.0
    rot = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    trans = np.zeros((3, 1))
    cnt = 0
    while threshold > 0.001 and cnt < 10:
        scale1, rot1, trans1, max_dist = icpIter(data, gt, rot * scale, trans, threshold)
        scale = scale * scale1
        rot = rot1 @ rot
        trans = rot1 @ trans + trans1
        threshold = max_dist * 0.5
        cnt += 1
    result = np.zeros((4, 4), dtype = float)
    result[0:3, 0:3] = rot
    result[0:3, 3:4] = trans
    result[3, 3] = 1
    colors1, colors2 = [], []
    for query_pt, value_pt in zip(data, gt):
        query_pt = rot @ query_pt + trans
        colors1.append([0., 1., 0.])
        colors2.append([0., 0., 1.])
    return scale, result, colors1, colors2

def transform_scale(scale, trans, pos):
    result = np.zeros((4, 4), dtype = float)
    result[0:3, 0:3] = trans[0:3, 0:3] @ pos[0:3, 0:3]
    result[0:3, 3:4] = scale * trans[0:3, 0:3] @ pos[0:3, 3:4] + trans[0:3, 3:4]
    result[3, 3] = 1
    return result

if __name__ == "__main__":
    scene = 'Easyship'
    file_path1 = f'../data/Jrender_Dataset/{scene}/transforms_train_ori.json'
    file_path2 = f'../data/Jrender_Dataset/{scene}/transforms_train_ngp.json'
    file_path3 = f'../data/Jrender_Dataset/{scene}/transforms_B_test_ori.json'
    with open(file_path1,'r')as f:
        data1=json.load(f)
    with open(file_path2,'r')as f:
        data2=json.load(f)
    with open(file_path3,'r')as f:
        data3=json.load(f)
    data = []
    gt_m = []
    gt = []
    frame2_map = {}
    for frame2 in data2['frames']:
        matrix2 = np.array(frame2['transform_matrix'], np.float32)
        frame2_map[frame2['file_path']] = matrix2
    for frame1 in data1['frames']:
        matrix1 = matrix_nerf2ngp(np.array(frame1['transform_matrix'], np.float32))
        matrix2 = frame2_map[frame1['file_path']]
        data.append(matrix1[0:3, 3:4])
        gt.append(matrix2[0:3, 3:4])
        gt_m.append(matrix2)
    scale, avg_trans, colors1, colors2 = icp(data, gt)
    print(scale)
    print(avg_trans)

    data_trans_m = []
    data_trans = []
    for frame1 in data1['frames']:
        matrix1 = transform_scale(scale, avg_trans, matrix_nerf2ngp(np.array(frame1['transform_matrix'], np.float32)))
        data_trans.append(matrix1[0:3, 3:4])
        data_trans_m.append(matrix1)
    data_test = []
    data_test_fix = []
    color_test = []
    color_test_fix = []
    for frame1 in data3['frames']:
        matrix1 = transform_scale(scale, avg_trans, matrix_nerf2ngp(np.array(frame1['transform_matrix'], np.float32)))[0:3, 3:4]
        data_test.append(matrix1)
        color_test.append([1., 1., 0.])
        color_test_fix.append([1., 0., 1.])
        min_dist = np.linalg.norm(matrix1 - data_trans[0])
        delta2_t = delta_t = gt[0] - data_trans[0]
        min_dist2 = min_dist * 2
        for dt, g in zip(data_trans, gt):
            d = np.linalg.norm(matrix1 - dt)
            if d < min_dist:
                min_dist = d
                delta_t = g - dt
            elif d < min_dist2:
                min_dist2 = d
                delta2_t = g - dt
        matrix1_fix = matrix1 + (delta_t + delta2_t) / 2
        data_test_fix.append(matrix1_fix)

    if False:
        data_trans = np.asarray(data_trans).squeeze()
        data_test = np.asarray(data_test).squeeze()
        data_test_fix = np.asarray(data_test_fix).squeeze()
        gt = np.asarray(gt).squeeze()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data_trans)
        pcd.colors = o3d.utility.Vector3dVector(colors1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(gt)
        pcd2.colors = o3d.utility.Vector3dVector(colors2)
        pcd3 = o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(data_test)
        pcd3.colors = o3d.utility.Vector3dVector(color_test)
        pcd4 = o3d.geometry.PointCloud()
        pcd4.points = o3d.utility.Vector3dVector(data_test_fix)
        pcd4.colors = o3d.utility.Vector3dVector(color_test_fix)
        lines = o3d.geometry.LineSet()
        lines_index = []
        for i in range(gt.shape[0]):
            lines_index.append([i, i+gt.shape[0]])
        lines.lines = o3d.utility.Vector2iVector(lines_index)
        lines.points = o3d.utility.Vector3dVector(np.concatenate((data_trans, gt), axis=0))
        o3d.visualization.draw_geometries([pcd, pcd2, pcd3, pcd4, lines])
    else:
        # file_path_vals = [f'../data/Jrender_Dataset/{scene}/transforms_val',
        #                   f'../data/Jrender_Dataset/{scene}/transforms_test']
        file_path_vals = [f'../data/Jrender_Dataset/{scene}/transforms_B_test']
        for file_path_val in file_path_vals:
            with open(file_path_val + '_ori.json','r')as f:
                data_val=json.load(f)
            for frame in data_val['frames']:
                matrix = transform_scale(scale, avg_trans, matrix_nerf2ngp(np.array(frame['transform_matrix'], np.float32)))
                m_trans = matrix[0:3, 3:4]

                delta2_t = delta_t = gt[0] - data_trans[0]
                delta2_r = delta_r = gt_m[0][0:3, 0:3] @ np.linalg.inv(data_trans_m[0][0:3, 0:3])
                min_dist = np.linalg.norm(m_trans - data_trans[0])
                min_dist2 = min_dist * 2
                for dt, g in zip(data_trans_m, gt_m):
                    d = np.linalg.norm(m_trans - dt[0:3, 3:4])
                    if d < min_dist:
                        min_dist = d
                        delta_t = g[0:3, 3:4] - dt[0:3, 3:4]
                        delta_r = g[0:3, 0:3] @ np.linalg.inv(dt[0:3, 0:3])
                    elif d < min_dist2:
                        min_dist2 = d
                        delta2_t = g[0:3, 3:4] - dt[0:3, 3:4]
                        delta2_r = g[0:3, 0:3] @ np.linalg.inv(dt[0:3, 0:3])
                matrix[0:3, 3:4] = m_trans + (delta_t + delta2_t) / 2
                q1 = quat.from_rotation_matrix(delta_r)
                q2 = quat.from_rotation_matrix(delta2_r)
                q_fix = quat.slerp(q1, q2, 0, 1, 0.5)
                u, s, v = np.linalg.svd(quat.as_rotation_matrix(q_fix) @ matrix[0:3, 0:3])
                matrix[0:3, 0:3] = u @ v
                frame['transform_matrix'] = (matrix).tolist()
            with open(file_path_val + '_ngp.json', "w") as outfile:
                json.dump(data_val, outfile, indent=4)
