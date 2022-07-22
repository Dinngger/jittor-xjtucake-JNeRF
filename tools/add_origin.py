'''
---------------------------------------------------------------------------------------------------------------------
Copyright (c) 1986 - 2022, AMA team, Institute of AI and Robotics, Xi'an Jiaotong University. Proprietary and
Confidential All Rights Reserved.
---------------------------------------------------------------------------------------------------------------------
NOTICE: All information contained herein is, and remains the property of AMA team, Institute of AI and Robotics,
Xi'an Jiaotong University. The intellectual and technical concepts contained herein are proprietary to AMA team, and
may be covered by P.R.C. and Foreign Patents, patents in process, and are protected by trade secret or copyright law.

This work may not be copied, modified, re-published, uploaded, executed, or distributed in any way, in any time, in
any medium, whether in whole or in part, without prior written permission from AMA team, Institute of AI and
Robotics, Xi'an Jiaotong University.

The copyright notice above does not evidence any actual or intended publication or disclosure of this source code,
which includes information that is confidential and/or proprietary, and is a trade secret, of AMA team.
---------------------------------------------------------------------------------------------------------------------
FilePath: /tools/add_origin.py
Author: Dinger
Author's email: dinger@stu.xjtu.edu.cn
'''
import os
import json

scenes = ["Car", "Coffee", "Easyship", "Scar", "Scarf"]

# for scene in scenes:
scene = "Scarf"
for back in ["test", "train", "val"]:
    file_path = f'../data/Jrender_Dataset/{scene}/transforms_{back}_ori.json'
    file_path_ori = f'../data/Jrender_Dataset/{scene}/transforms_{back}.json'
    os.rename(file_path, file_path_ori)
