# -cnn-

车载毫米波雷达宽带干扰抑制实验结果。

当前上传内容位于 `radar_tf_denoise/`：

- `v0：单通道幅度时频图 U-Net`
- 输入：带干扰复数雷达信号 `sb` 的 STFT 幅度图
- 标签：无干扰干净信号 `sb0` 的 STFT 幅度图
- 输出：模型预测的干净时频图

主要结果文件：

- `radar_tf_denoise/checkpoints/best_model.pth`：当前 v0 最优模型权重
- `radar_tf_denoise/outputs/metrics.csv`：200 条测试样本逐样本指标
- `radar_tf_denoise/outputs/summary_metrics.csv`：整体汇总指标
- `radar_tf_denoise/outputs/case_analysis/`：典型成功/失败样本分析图
- `radar_tf_denoise/outputs/case_analysis/case_summary.csv`：案例样本指标汇总

本仓库不包含原始 ARIM `.mat` / `.npy` 数据文件。
