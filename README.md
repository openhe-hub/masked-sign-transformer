## Introduction
## ChangeLog
### V0: Based on MAE + Transformer
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
  5. total_variation_loss 
### V3
* add body bone subset info
* bone render while inference
* add body_bone_length_loss
### V4: add tokenizer (Vanilla STTF-like)
* keypoint tokenizer, instead of flattern one
* spatial-temporal embedding, instead of position embedding
### V5: add asymmetric encoder-decoder (large-small)
* MAE-like asymmetric encoder-decoder
* a fixed total ratio of spatial-temporal mask
### V6: scale up transformer size

## Future Plan
### Bert-based: double direction context input
### TAE-based: stream chatsign generation
### Video Part: SVD / GAN ?