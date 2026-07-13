# 风速时间序列预测 - 机器学习课程项目

## 项目简介
本项目基于10m、50m、100m三个高度的气象传感器实测数据，构建风速时间序列预测模型，完成单步与多步预测任务。项目实现了线性回归、LSTM、Transformer、CNN1D四类对比模型，集成Optuna贝叶斯超参数优化，严格遵循时序数据7:2:1顺序划分规则，特征工程与归一化均基于训练集统计量计算，全程杜绝数据泄露。

## 预测任务
1. **单步预测（Single-step）**：基于8小时历史窗口，预测下一时刻风速
2. **多步预测A（Multi-step_A_1h）**：基于8小时历史窗口，预测未来1小时（6个时间步）风速
3. **多步预测B（Multi-step_B_16h）**：基于8小时历史窗口，预测未来16小时（96个时间步）风速

## 环境依赖
- 运行环境：Python 3.8 及以上版本
- 一键安装依赖：在项目**根目录**下执行以下命令
  ```bash
  pip install -r requirements.txt
  ```

## 数据集说明
数据集来源于 Hugging Face，包含三个高度的气象观测时序数据：
- 10m高度数据集：https://huggingface.co/datasets/Antajitters/WindSpeed_10m
- 50m高度数据集：https://huggingface.co/datasets/Antajitters/WindSpeed_50m
- 100m高度数据集：https://huggingface.co/datasets/Antajitters/WindSpeed_100m

脚本默认自动在线下载数据集；国内网络环境下可能出现连接超时，可通过配置国内镜像源解决，详见运行步骤。

## 运行步骤
1. 打开命令行窗口，进入项目根目录后，切换到代码与模型所在目录：
   ```bash
   cd project
   ```

2. （国内网络环境必选）配置 Hugging Face 国内镜像源，解决下载超时问题：
   - Windows CMD 环境：
     ```bash
     set HF_ENDPOINT=https://hf-mirror.com
     ```
   - Windows PowerShell 环境：
     ```bash
     $env:HF_ENDPOINT = "https://hf-mirror.com"
     ```
   > 该配置仅对当前命令行窗口生效，关闭后失效。

3. 执行主脚本启动程序：
   ```bash
   python 代码.py
   ```

## 运行模式说明
代码顶部的 `RUN_MODE` 参数可切换两种运行模式：
- **EVAL（默认模式）**：快速评估模式。自动加载同目录下已训练完成的 `.pth` 模型权重，直接在测试集上完成预测，输出四项评估指标并生成可视化结果，运行速度快，推荐用于结果验证。
- **TUNE 模式**：完整训练模式。启动Optuna贝叶斯超参数搜索，从头训练所有模型，输出最优参数与最终权重文件，耗时较长，用于完整复现训练过程。

## 目录结构
```
WindSpeed-Prediction/
├── project/                # 核心运行目录
│   ├── 代码.py             # 主程序脚本（含数据加载、清洗、特征工程、模型定义、训练、评估、可视化全流程）
│   ├── *.pth               # 所有任务训练好的模型文件（同时保存超参数与网络权重）
│   └── *.png / *.csv       # 程序运行生成的可视化图表、指标结果表格
├── README.md               # 项目说明文档
└── requirements.txt        # Python 依赖库清单
```

## 评估指标
在测试集上采用四项标准指标量化模型预测性能：
- **MSE（均方误差）**：预测值与真实值误差的平方均值，数值越小精度越高
- **RMSE（均方根误差）**：均方误差的平方根，量纲与原始风速一致，便于直观理解
- **MAE（平均绝对误差）**：预测值与真实值绝对误差的均值，对异常值鲁棒性更强
- **R²（决定系数）**：衡量模型对数据变异的解释程度，取值越接近1表示拟合效果越好

## 常见问题
1. **数据集下载连接超时**：执行上述国内镜像配置命令后重新运行即可。
2. **提示找不到 .pth 模型文件**：请确保在 `project` 目录下执行脚本，保证相对路径匹配。
3. **PyTorch 安装失败**：可通过清华镜像源单独安装PyTorch CPU版本，命令如下：
   ```bash
   pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
4. **训练模式运行过慢**：TUNE模式包含20轮贝叶斯搜索+完整训练，耗时较久；验证结果建议使用默认EVAL模式。
