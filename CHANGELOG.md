## 本项目用于参加第二届Jittor人工智能挑战赛赛道二。

## 该仓库基于JNeRF。有如下改进：
- 改进了训练脚本和显示方式，修改数据集加载的位置，减少显存占用。
- 增加了模型查看GUI，基于dearpygui。
- 增加了数据集json处理脚本，利用icp和插值算法修正数据集中的位姿错误。
- 复现了RefNeRF。请切换git分支查看。
- 调节参数以提升模型效果。

## 使用方法
1. 修改projects/ngp/configs/ngp_comp.py中的exp_name以运行不同的场景。
2. 修改run_comp.sh中的task以执行不同的任务。
   - train执行训练任务，训练时只显示loss，训练完成后会加载验证集，输出平均最小和最大的psnr。
   - test会渲染输出测试集图片。
   - val_all加载验证集，输出平均最小和最大的psnr。
   - gui启动图形界面，可以自由查看模型。
