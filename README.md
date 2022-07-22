# jittor-xjtucake-JNeRF
# jittor 人工智能算法挑战赛可谓渲染新视角生成赛题 洗脚点心队

# 简介
本项目包含了第二届计图挑战赛-可微渲染新视角生成比赛的代码实现。该项目是本次比赛A榜Top3，B榜Top2.
本项目基于JNeRF开发，其特点是：
1. 通过MVS方法获取较为准确的相机参数，使得Car和Easyship场景下更好的结果；
2. 修改渲染方程，改为6通道rgb渲染，将反射光和透射光分离，大幅提升NeRF对透明物体的渲染效果；
3. 调整aabb，scale，offset等参数，使得物体占据aabb中的最优位置，提升细节表现力。

另外，我们还做了一些小更改：
- 一些脚本
- gui

# 安装
参考JNeRF