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

### 训练脚本（`src/bridge/train_bridge.py`）
- DataLoader 直接使用新的 dataset。
- Loss 只在 gap 段计算，可复用原有的 Huber、velocity、accel、TV 权重。
- 未来如需骨骼/其它正则，可在 gap 段上扩展。

### 推理可视化（`src/bridge/inference_bridge.py`）
- 支持加载训练好的 `PoseBridgeTransformerV6`，并自动恢复 `norm_params` / `subset`，生成原像素坐标的重建序列。
- 输出视频默认展示 “Ground Truth vs Predicted” 两列，且在预测列中用红色高亮 `gap_length` 帧的过渡段，方便快速对比模型填补效果。
- 运行示例：
  ```bash
  python src/bridge/inference_bridge.py --checkpoint checkpoints/exp_bridge_epoch_100.pth \
      --output_dir output/bridge_vis --num_samples 4 --start_index 0 --fps 5
  ```

### 双段桥接推理（`src/bridge/inference_bridge_pair.py`）
- 针对“给定两个独立 motion，生成中间过渡”的场景：脚本会从 `motion_a` 提取 `pre_context_length` 帧、从 `motion_b` 提取 `post_context_length` 帧，将二者输入 bridge 模型以推理 gap。
- 支持直接填写 `.pkl` 的绝对/相对路径，或仅填文件名（默认会去 `config.data.data_dir` 下查找）。可用 `--start_a`、`--start_b` 控制各自的起始帧位置（默认 A 取末尾、B 取开头）。
- 输出视频包含三列：`Context Only`（pre/gap/post）、`Bridge Result`（预测 gap，红色高亮）、`Full A+Bridge+B`（完整的 motion A + gap + motion B，方便直接观看完整过渡）。若加上 `--save_pkl`，会把该完整序列渲染成 `pose_pixels`（与 mm-style 一致）并连同 `motion_a/b` 的原始 pkl 内容一并保存，便于下游直接生成视频或做进一步处理。
- 如果只想保留完整序列一列，可以加 `--full_only`，这样 MP4 就只会输出 `Predicted (A + Bridge + B)` 这一列。
- 运行示例：
  ```bash
  python src/bridge/inference_bridge_pair.py \
      --checkpoint checkpoints/exp_bridge_epoch_100.pth \
      --motion_a 01abcde123_kps.pkl \
      --motion_b 09vwxyz999_kps.pkl \
      --start_a -1 --start_b 0 \
      --output_dir output/bridge_pair_vis --fps 6 --save_pkl
  ```
- 批量模式：提供 `--pairs_file pairs.csv` 时可一次性运行多组，文件以逗号分隔，格式为  
  `motion_a,motion_b,start_a,start_b,fps,save_pkl`（后四个字段可空）；例如：
  ```
  007diq0k5t_kps.pkl,00mvc2hvbq_kps.pkl,-1,0,5,true
  00dfosh03o_kps.pkl,00fyz5gaeq_kps.pkl,10,0,6,
  ```
  执行 `python src/bridge/inference_bridge_pair.py --checkpoint ... --pairs_file pairs.csv --output_dir ...`
  即可逐条生成。

> 该策略和模型文件独立于现有 V4 pipeline，避免影响旧实验；需要桥段预测时，使用 `src/bridge/train_bridge.py` & `PoseBridgeTransformerV6` 即可。
