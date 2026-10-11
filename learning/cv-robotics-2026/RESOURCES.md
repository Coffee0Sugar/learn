# GitHub 项目与学习文档

[返回计划](README.md) · [项目任务书](PROJECT.md) · [进度清单](PROGRESS.md)

整理日期：2026-10-11。先按周使用必修资料，选学只在项目或岗位有需要时打开。下列链接为上游项目和官方文档；网页可访问不代表本机已完成安装测试。实际使用时记录 release／commit 和依赖组合。ROS 文档访问状态见 R09。

## 使用顺序

| 优先级 | 项目 | 学习周次 | 使用范围 |
| --- | --- | --- | --- |
| 必修 | [R01 TorchVision](#r01) | W01—W02、W05、W09 | 检测训练、数据、评估；已有熟悉的框架可沿用 |
| 必修 | [R02 OpenCV](#r02) | W03、W05—W06、W09 | 相机几何、坐标与深度处理 |
| 按需查底层 | [R03 MuJoCo](#r03) | W01—W04 | 理解仿真状态、步进和渲染 |
| 必修任务层 | [R04 robosuite](#r04) | W01—W10 | 现成机械臂、任务、相机与控制器 |
| 必修选章 | [R05 Modern Robotics](#r05) | W03—W04 | 刚体变换与正逆运动学 |
| 必修 | [R06 ONNX Runtime](#r06) | W11—W12 | 导出验证、推理与 C++ 接入 |
| 持续练习 | [R07 C++ Core Guidelines](#r07) | W01—W16 | 对照本人代码学习资源和接口管理 |
| 选学 | [R08 Open3D](#r08) | W09 起 | 点云可视化与过滤，仅在定位需要时引入 |
| 岗位驱动选学 | [R09 ROS 2](#r09) | W13，或按实际岗位前移 | 模块通信和坐标接口，不扩展到完整导航 |
| 主线完成后选学 | [R10 LeRobot／ACT](#r10) | W12 以后另行评估 | 机器人学习岗位的后续路线 |
| 导航／备用 | [R11 中文知识地图与模型库](#r11) | 按需 | 查概念或模型，不额外增加完整课程 |

<a id="r01"></a>
## R01 · TorchVision：检测训练与评估

- GitHub：[pytorch/vision](https://github.com/pytorch/vision)
- 官方教程：[Object Detection Finetuning Tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- 教程代码仓库：[pytorch/tutorials](https://github.com/pytorch/tutorials)

阅读范围：数据集接口、标注格式、数据增强、模型替换、训练与验证。官方教程包含实例分割内容，本计划只取检测流程所需部分。熟悉其他检测框架时可保留原框架，不要求为换工具重做项目。

练习：先恢复一个已有模型的评估，再做一次小规模微调；保存输入、标签、预测和错误样例。解释 IoU／AP／Recall、置信度阈值和 NMS 的不同作用。

验收：能从新环境按说明重跑，知道数据如何划分，能解释一个漏检案例。后续导出时先做小样本验证；检测模型的动态输出或算子支持可能需要额外处理，不假定所有模型都能一键导出。

<a id="r02"></a>
## R02 · OpenCV：相机、几何与三维定位

- GitHub：[opencv/opencv](https://github.com/opencv/opencv)
- 官方文档：[Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

阅读范围：针孔模型、内外参、投影、畸变、旋转和平移表示；按需要阅读相机标定、PnP 相关接口。固定相机外参转换先做通，手眼标定理论可作为后续补充。

练习：已知三维点投影到图像，再结合深度恢复三维位置；验证变换的逆和复合。写清矩阵表达的是“从哪个坐标系到哪个坐标系”。

验收：定位结果有单位与误差报告；能说明为什么检测框中心不必然是合适的抓取点。仿真提供的深度可能不是以米表示的相机 Z 值，必须先按仿真接口语义转换。

<a id="r03"></a>
## R03 · MuJoCo：理解仿真底层

- GitHub：[google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)
- 官方文档：[Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)

阅读范围：模型与运行状态、仿真步进、查看器和渲染。通过一个最小示例理解观测和动作，不从零编写整套机器人环境。

练习：运行示例、读取关节状态、保存一帧图像。先记录系统与渲染后端，避免把环境故障误判为控制算法故障。

验收：能解释仿真一步做了什么，以及视觉频率与控制频率的区别。

<a id="r04"></a>
## R04 · robosuite：主项目任务与控制器

- GitHub：[ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite)
- [官方概览](https://robosuite.ai/docs/overview.html)
- [安装与平台说明](https://robosuite.ai/docs/installation.html)
- [环境与任务接口](https://robosuite.ai/docs/modules/environments.html)
- [控制器说明](https://robosuite.ai/docs/modules/controllers.html)

阅读范围：安装验证、创建环境、观察与动作、相机、控制器配置。第一版使用一个已有机械臂和简单 Lift 任务；抬升稳定后才拓展 PickPlace 或受控放置。任务和控制器的实际配置以锁定版本的文档为准。

练习：随机动作演示只用于验证安装；随后明确动作空间，用已有控制器实现脚本接近、闭合、抬升。逐步接入 RGB-D 和检测结果。

验收：能区分物体真实位姿诊断基线与最终视觉输入方案；能复现指定种子的场景和失败过程。该框架提供任务与控制器，不自动提供本项目的完整视觉抓取方案。

<a id="r05"></a>
## R05 · Modern Robotics：只学当前需要的章节

- GitHub：[NxRLab/ModernRobotics](https://github.com/NxRLab/ModernRobotics)
- 作者课程与教材入口：[Modern Robotics Book Resources](https://modernrobotics.northwestern.edu/nu-gm-book-resource/)

阅读范围：第 3 章刚体运动，第 4 章正运动学，第 6 章逆运动学；雅可比按理解逆运动学所需补读。动力学、最优控制不列入当前必修。

练习：Python 实现刚体变换的复合与求逆，用库中的示例交叉检查；理解逆运动学存在不可达和多解情况。

验收：能画坐标关系并说明末端目标如何变成关节动作。仓库定位是教学代码，不能直接把它的例程当作经过工程验证的控制系统。

<a id="r06"></a>
## R06 · ONNX Runtime：模型部署与性能评估

- GitHub：[microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
- 示例：[microsoft/onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples)
- [PyTorch ONNX 导出入口](https://docs.pytorch.org/docs/stable/onnx.html)
- [ONNX Runtime C++ 文档](https://onnxruntime.ai/docs/get-started/with-cpp.html)
- [模型优化文档](https://onnxruntime.ai/docs/performance/model-optimizations/)

阅读范围：导出输入／输出约定、模型加载、张量数据、运行时后端、C++ Session；先做基本部署，再按测量结果决定是否优化。

练习：先比较 PyTorch 与 ONNX 输出，再检查最终检测指标；按预处理、模型推理、后处理拆分时间；通过同一组图片核对 Python 和 C++ 路径。

验收：报告硬件、输入大小、batch、精度、线程／后端、预热次数和 P50／P95 延迟。优化后不能只报速度，也要报效果变化。没有 NVIDIA GPU 时不把 TensorRT 设置为必修。

<a id="r07"></a>
## R07 · C++ Core Guidelines：围绕自己的代码复习

- GitHub：[isocpp/CppCoreGuidelines](https://github.com/isocpp/CppCoreGuidelines)
- 阅读入口：仓库中的 CppCoreGuidelines.md。

阅读范围：资源管理、接口、函数、类、错误处理；并发部分按面试和项目需要补读。

练习：检查推理模块的对象生命周期、张量缓冲区、智能指针和异常路径；写一个能构建运行的模块。每周两道基础编程题，要求解释复杂度与边界条件。

验收：能说明一次内存／资源问题如何定位，能解释为什么选择引用、值传递或智能指针。不要把背诵条款当作掌握 C++。

<a id="r08"></a>
## R08 · Open3D：点云处理选学

- GitHub：[isl-org/Open3D](https://github.com/isl-org/Open3D)
- 官方教程：[Point cloud](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html)

只在 RGB-D 定位需要时学习点云可视化、下采样和基础处理；围绕目标区域深度做小练习。不将完整三维重建或 SLAM 加入本轮计划。

验收：能可视化目标区域并解释噪声、离群点、可见表面中心与物体几何中心的区别。简单深度处理足够解决问题时，不必新增这个依赖。

<a id="r09"></a>
## R09 · ROS 2：岗位驱动的接口选学

- GitHub 示例：[ros2/examples](https://github.com/ros2/examples)
- 官方文档入口：[Jazzy Tutorials](https://docs.ros.org/en/jazzy/Tutorials.html)
- 文档源码备用：[ros2/ros2_documentation](https://github.com/ros2/ros2_documentation)

核对说明：官方网页在本次自动访问中触发了反爬验证；保留官方入口并提供源码仓库，不能将验证页当成教程正文。浏览器可访问时按相应发行版继续；源码仓库选择与所用发行版一致的分支。

阅读范围：节点、topic、service、action 的区别，以及 publisher／subscriber 示例；坐标接口需要时再学 tf2。

练习：发送目标位置并读取状态，给消息标注坐标系与时间。只做必要桥接，不为使用 ROS 2 重写已完成的仿真项目。

验收：解释数据流、消息频率和异常输入处理。安装前单独确认发行版与操作系统支持。

<a id="r10"></a>
## R10 · LeRobot／ACT：机器人学习方向的后续扩展

- GitHub：[huggingface/lerobot](https://github.com/huggingface/lerobot)
- 官方教程：[ACT](https://huggingface.co/docs/lerobot/main/en/act)

启用条件：主项目已完成评估、部署、文档，且目标岗位确实要求模仿学习；通常另开后续学习阶段。先理解观测、动作、动作分块、数据和闭环评估，再决定是否复现。

不要假定当前 robosuite 任务、数据格式和 LeRobot 训练入口可以直接拼接。必须先检查环境适配、观测／动作空间、数据格式和预算。

当前视觉＋规则抓取项目不证明 ACT／VLA 研发能力；没有做过的内容不写入简历。

<a id="r11"></a>
## R11 · 知识地图与备用模型

- 中文导航：[datawhalechina/dive-into-embodied-ai](https://github.com/datawhalechina/dive-into-embodied-ai)
- [中文学习路径](https://github.com/datawhalechina/dive-into-embodied-ai/blob/master/docs/overview/learning-path.md)
- 备用模型库：[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)

中文材料用于查概念与方向；模型库用于查看已有机器人模型。已有 robosuite 机械臂能完成任务时无需另加模型库。学习目录中的行业判断不能替代实际社招职位要求。

## 资源核对与引用规则

- 本清单只链接项目，不自动克隆全部依赖，不复制整篇教程。
- 开始使用某项目时记录其版本、许可证、权重和数据的来源；代码、权重、数据的许可可能分别不同。
- 若链接变动，从同一项目的官方仓库或文档入口找对应章节，并更新此清单。
- 学习“完成”必须附代码、图、报告或问答记录，不能仅凭收藏链接打勾。
