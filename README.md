## Stage 1 (Single Direction): Based on MAE + Transformer
### V0
* 2d kps feature (x, y)
* position embedding
* spatial-temporal mask
### V1
* 3d kps feature (x, y, conf)
* filter low conf segment
* add low-conf mask
### V2
* add multi loss
  1. reconstruction_loss
  2. full_sequence_reconstruction_loss
  3. velocity_consistency_loss
  4. acceleration_consistency_loss
  5. bone_length_consistency_loss
  6. total_variation_loss 