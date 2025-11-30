## Bridge Masking Strategy

This document tracks the contiguous-gap masking variant that powers the new bridge model/dataset pair.

### Motivation
- Random逐点mask很难让网络学会“两个片段之间的过渡”。
- 新策略只隐藏连续 `n` 帧（`config.masking.gap_length`），模型需要根据 gap 前后的上下文来预测中间的关键点运动，包括 `(x, y, conf)`。

### Dataset 侧（`PoseBridgeDatasetV6`）
1. 依旧读取 `config.data.sequence_length` 的滑动窗口以及分部件的 `(T, V, 3)` 数据。
    2. `config.masking` 中新增 `pre_context_length` 与 `post_context_length`，窗口的长度由 `pre + gap + post` 精确控制（例如 5 + 6 + 5 = 16）。
    3. DataLoader 现在会在加载时自动把 `keypoints` 拆成 `body/left/right/face` 四个分块（若源数据已经是 per-part dict 也能兼容）。
    4. Dataset 直接切出固定长度的 pre/gap/post 段；若未来需要 padding，可以继续使用 mask 字段（目前恒为 False）。
    5. 将 gap 的 GT 作为 `gap_target`，供 loss 使用。

### Model 侧（`PoseBridgeTransformerV6`）
- 结构沿用 per-part 处理思路：每个部件都有独立的编码-解码模块。
- 编码器分别接收 `pre` / `post` 片段，拼接后通过 Transformer Encoder，mask 会阻止 padding 参与注意力。
- Decoder 使用长度为 `gap_length` 的 learnable queries，输出 gap 的 `(x, y, conf)`，再通过线性层映射回每个关键点。

### 训练脚本（`train_bridge.py`）
- DataLoader 直接使用新的 dataset。
- Loss 只在 gap 段计算，可复用原有的 Huber、velocity、accel、TV 权重。
- 未来如需骨骼/其它正则，可在 gap 段上扩展。

### 推理可视化（`inference_bridge.py`）
- 支持加载训练好的 `PoseBridgeTransformerV6`，并自动恢复 `norm_params` / `subset`，生成原像素坐标的重建序列。
- 输出视频默认展示 “Ground Truth vs Predicted” 两列，且在预测列中用红色高亮 `gap_length` 帧的过渡段，方便快速对比模型填补效果。
- 运行示例：
  ```bash
  python src/inference_bridge.py --checkpoint checkpoints/exp_bridge_epoch_100.pth \
      --output_dir output/bridge_vis --num_samples 4 --start_index 0 --fps 5
  ```

> 该策略和模型文件独立于现有 V4 pipeline，避免影响旧实验；需要桥段预测时，使用 `train_bridge.py` & `PoseBridgeTransformerV6` 即可。
