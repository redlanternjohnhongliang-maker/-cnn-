# Radar TF Denoising Experiments

这个仓库只上传给 GPT/代码审查用的最小内容：代码、训练日志、CSV 指标结果。

没有上传：

- ARIM 原始数据：`*.mat`、`*.npy`
- 模型权重：`*.pth`
- 可视化图片：`*.png`、`*.zip`

主要代码在 `radar_tf_denoise/`。

## 版本路线

- `v0`：单通道幅度时频图 U-Net，普通 L1。
- `v0.1`：单通道幅度时频图 U-Net，加权 L1。
- `v1`：复数两通道 U-Net。
- `v1.1`：复数两通道 U-Net，加权幅度损失。
- `CFAR-Z / CFAR-AC`：传统 CFAR 基线。
- `hybrid_v11_cfar`：CFAR mask 区域用 v1.1，非 mask 区域保留 noisy。
- `soft_hybrid_search`：搜索 `rho / pfa / dilation_iter`。
- `v1.2`：CFAR-mask gated residual U-Net。
- `v1.3`：v1.2 加 mask 内修复损失。

## 关键脚本

- `radar_tf_denoise/train.py`：训练入口。
- `radar_tf_denoise/evaluate.py`：评估入口。
- `radar_tf_denoise/data_loader.py`：STFT 数据读取和 `complex_mask_residual` 模式。
- `radar_tf_denoise/losses.py`：`complex_weighted_mag` 和 `complex_weighted_mag_mask`。
- `radar_tf_denoise/baselines_cfar.py`：CFAR-Z / CFAR-AC。
- `radar_tf_denoise/soft_hybrid_search.py`：soft hybrid 参数搜索。
- `radar_tf_denoise/analyze_v12_error_source.py`：mask 内/外误差来源分析。

## 主要结果

关键汇总 CSV：

- `radar_tf_denoise/outputs_v11/summary_metrics.csv`
- `radar_tf_denoise/outputs_soft_hybrid_search/best_summary.csv`
- `radar_tf_denoise/outputs_v12_full/summary_metrics.csv`
- `radar_tf_denoise/outputs_v13_full/summary_metrics.csv`
- `radar_tf_denoise/outputs_v12_error_analysis/error_source_summary.csv`
- `radar_tf_denoise/outputs_v13_error_analysis/error_source_summary.csv`

核心对比：

| 方法 | spectrum改善 | 改善样本率 | 目标峰保持 | 目标峰误差 | 目标峰改善率 | 底噪比 |
|---|---:|---:|---:|---:|---:|---:|
| v1.1 | 0.939570 | 0.810000 | 0.958368 | 61.293633 | 0.665000 | 1.045906 |
| best hybrid | 0.933625 | 0.935000 | 0.940215 | 59.308798 | 0.810000 | 1.043306 |
| v1.2 full | 0.909427 | 0.940000 | 0.955796 | 50.505675 | 0.840000 | 1.211985 |
| v1.3 full | 0.899663 | 0.945000 | 0.953293 | 52.758024 | 0.845000 | 1.269377 |

当前结论：

- v1.2 的目标峰误差最好，但底噪偏高。
- 误差归因显示 v1.2 的问题主要在 mask 内修复不够干净。
- v1.3 直接加 `mask_loss_weight=5.0` 后，mask 内 MAE 反而升高，底噪也更高，说明这个权重/损失设计不理想。

## 数据路径说明

本地训练时使用：

- `training/arim_train_random.npy`
- `training/arim_test_random.npy`

这些文件未上传 GitHub，需要本地生成或放回对应目录。
