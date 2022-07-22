from ast import parse
from time import sleep
import jittor as jt
# from train import Trainer
# from model import NerfNetworks, HuberLoss
from tqdm import tqdm
# from utils.dataset import  NerfDataset
import argparse
import numpy as np
import os
import sys
import importlib

sys.path.append('./python')
from jnerf.runner import Runner 
from jnerf.utils.config import init_cfg
jt.flags.use_cuda = 1
assert jt.flags.cuda_archs[0] >= 61, "Failed: Sm arch version is too low! Sm arch version must not be lower than sm_61!"
init_cfg('./projects/ngp/configs/ngp_comp.py')
runner = Runner()
# runner.test(True, True)

print(os.popen("sed -i '41s/0/1/g' ./projects/ngp/configs/ngp_comp.py"))
sleep(1)
init_cfg('./projects/ngp/configs/ngp_comp.py')
runner = Runner()
# runner.test(True, True)
print(os.popen("sed -i '41s/1/3/g' ./projects/ngp/configs/ngp_comp.py"))
sleep(1)
init_cfg('./projects/ngp/configs/ngp_comp.py')
runner = Runner()
# runner.test(True, True)
print(os.popen("sed -i '41s/3/4/g' ./projects/ngp/configs/ngp_comp.py"))
sleep(1)
init_cfg('./projects/ngp/configs/ngp_comp.py')
runner = Runner()
# runner.test(True, True)

sys.path.remove('./python')
sys.path.append('./python_trans')
print(os.popen("sed -i '41s/4/2/g' ./projects/ngp/configs/ngp_comp.py"))
sleep(1)

need_del_m = []
for m in sys.modules.keys():
    if 'jnerf' in m:
        need_del_m.append(m)
for m in need_del_m:
    del sys.modules[m]

from jnerf.runner import Runner 
from jnerf.utils.config import init_cfg
jt.flags.use_cuda = 1
assert jt.flags.cuda_archs[0] >= 61, "Failed: Sm arch version is too low! Sm arch version must not be lower than sm_61!"
init_cfg('./projects/ngp/configs/ngp_comp.py')
runner = Runner()
runner.test(True, True)
print(os.popen("sed -i '41s/2/0/g' ./projects/ngp/configs/ngp_comp.py"))
sleep(1)
