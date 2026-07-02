# v0：单通道幅度时频图 U-Net

这个目录是独立实验目录，不依赖也不修改原始 `training` 训练入口。

当前版本命名为 `v0`：单通道幅度时频图 U-Net。

## 任务定义

输入：

```text
X = abs(STFT(sb))
```

标签：

```text
Y = abs(STFT(sb0))
```

其中：

`sb` 是带干扰复数雷达 beat signal，形状为 `(N, 1024)`。

`sb0` 是无干扰干净复数雷达 beat signal，形状为 `(N, 1024)`。

模型输入输出均为单通道时频图：

```text
输入: [B, 1, H, W]
输出: [B, 1, H, W]
```

默认 STFT 参数下，单条 1024 点信号输出：

```text
H = 256
W = 29
```

## 数据路径

训练数据：

```text
G:\雷达数据\arim-master\arim-master\training\arim_train.npy
```

测试数据：

```text
G:\雷达数据\arim-master\arim-master\training\arim_test.npy
```

所有路径都通过命令行参数传入，代码里不写死绝对路径。

## 随机切分数据

原始 `process.py` 固定取前 8000 条为测试集，剩下作为训练集。为了避免切分分布不均，可以使用本目录的随机切分脚本：

```powershell
cd "G:\雷达数据\arim-master\arim-master\radar_tf_denoise"
$env:PYTHONPATH="G:\雷达数据\arim-master\arim-master\radar_tf_denoise"
& "G:\Anaconda\envs\cnn_learn\python.exe" process_random_split.py `
  --mat_path "G:\雷达数据\arim-master\arim-master\arim_matlab\arim_smoke.mat" `
  --output_dir "G:\雷达数据\arim-master\arim-master\training" `
  --train_ratio 0.8 `
  --seed 707
```

输出文件：

```text
G:\雷达数据\arim-master\arim-master\training\arim_train_random.npy
G:\雷达数据\arim-master\arim-master\training\arim_test_random.npy
```

生成格式和原始 `process.py` 一致，字段为 `sb`、`sb0`、`amplitudes`、`distances`、`info_mat`。

## 训练

先用 200 条样本做小规模冒烟测试：

```powershell
cd "G:\雷达数据\arim-master\arim-master\radar_tf_denoise"
$env:PYTHONPATH="G:\雷达数据\arim-master\arim-master\radar_tf_denoise"
& "G:\Anaconda\envs\cnn_learn\python.exe" train.py `
  --train_path "G:\雷达数据\arim-master\arim-master\training\arim_train.npy" `
  --epochs 2 `
  --batch_size 8 `
  --lr 0.001 `
  --max_samples 200 `
  --device cuda
```

最优模型会保存到：

```text
radar_tf_denoise\checkpoints\best_model.pth
```

## 评估

```powershell
cd "G:\雷达数据\arim-master\arim-master\radar_tf_denoise"
$env:PYTHONPATH="G:\雷达数据\arim-master\arim-master\radar_tf_denoise"
& "G:\Anaconda\envs\cnn_learn\python.exe" evaluate.py `
  --test_path "G:\雷达数据\arim-master\arim-master\training\arim_test.npy" `
  --checkpoint "G:\雷达数据\arim-master\arim-master\radar_tf_denoise\checkpoints\best_model.pth" `
  --num_samples 4 `
  --max_samples 200 `
  --device cuda `
  --save_images
```

评估结果会保存到：

```text
radar_tf_denoise\outputs
```

量化指标会保存到：

```text
radar_tf_denoise\outputs\metrics.csv
```

每个样本会保存：

`带干扰时频图`

`干净时频图`

`模型输出时频图`

`带干扰、干净、模型输出三者的距离谱对比图`

注意：第一版模型只预测时频图幅度，不预测相位。距离谱对比中的模型输出信号使用“预测幅度 + 带干扰输入相位”近似重建，仅用于快速观察效果。

## 本机环境说明

在 Windows/Conda 环境下，`torch`、`scipy`、`matplotlib` 可能重复加载 OpenMP 运行库。`evaluate.py` 已经在脚本内设置 `KMP_DUPLICATE_LIB_OK=TRUE` 作为评估画图兜底。训练脚本没有设置这个变量。

评估脚本默认使用 `Agg` 后端保存图片，并优先使用 `Microsoft YaHei` / `SimHei` 显示中文标题。
