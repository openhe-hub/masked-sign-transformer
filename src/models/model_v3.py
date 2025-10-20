import torch
import torch.nn as nn
import math

from config_loader import config

class SpatioTemporalPositionalEncoding(nn.Module):
    """
    Spatio-Temporal Positional Encoding for pose sequences.
    This is identical to the one in model_v2.py.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000, n_kps=128):
        super(SpatioTemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Temporal Positional Encoding (fixed sine-cosine)
        pe_temporal = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_temporal[:, 0::2] = torch.sin(position * div_term)
        pe_temporal[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_temporal', pe_temporal.unsqueeze(0).unsqueeze(2))

        # Spatial Positional Encoding (learnable embedding)
        self.pe_spatial = nn.Embedding(n_kps, d_model)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of keypoint embeddings.
                        Shape: (batch_size, seq_len, n_kps, d_model)
        """
        B, T, K, D = x.shape
        x = x + self.pe_temporal[:, :T, :, :]
        x = x + self.pe_spatial.weight.unsqueeze(0).unsqueeze(0)
        return self.dropout(x)

class PoseTransformerV3(nn.Module):
    """
    Asymmetric Encoder-Decoder architecture based on MAE.
    - Encoder: A deep Transformer that processes ONLY the visible keypoint tokens.
    - Decoder: A shallow Transformer that reconstructs the full sequence from
               the encoded visible tokens and learnable mask tokens.
    """
    def __init__(self):
        super(PoseTransformerV3, self).__init__()
        
        # --- Shared Parameters ---
        n_kps = config['data']['n_kps']
        features_per_kp = 2
        d_model = config['model']['d_model']
        dropout = config['model']['dropout']
        
        # --- V2 Components (reused) ---
        self.keypoint_embedder = nn.Linear(features_per_kp, d_model)
        self.pos_encoder = SpatioTemporalPositionalEncoding(d_model, dropout, n_kps=n_kps)
        
        # --- Encoder ---
        encoder_d_model = d_model # Encoder can have its own dimension if needed
        encoder_n_head = config['model']['n_head']
        num_encoder_layers = config['model']['num_encoder_layers']
        encoder_dim_feedforward = config['model']['dim_feedforward']
        
        encoder_layer = nn.TransformerEncoderLayer(
            encoder_d_model, encoder_n_head, encoder_dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # --- Decoder ---
        decoder_d_model = config['model']['decoder_d_model']
        decoder_n_head = config['model']['decoder_n_head']
        num_decoder_layers = config['model']['num_decoder_layers']
        decoder_dim_feedforward = config['model']['decoder_dim_feedforward']

        # Projection from encoder's dimension to decoder's dimension
        self.encoder_to_decoder = nn.Linear(encoder_d_model, decoder_d_model)
        
        # Learnable token for masked positions in the decoder
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_d_model))

        decoder_layer = nn.TransformerEncoderLayer(
            decoder_d_model, decoder_n_head, decoder_dim_feedforward, dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers)

        # Final projection head
        self.output_head = nn.Linear(decoder_d_model, features_per_kp)
        
        self.d_model = d_model

    def forward(self, src, mask):
        """
        Args:
            src (Tensor): The masked input sequence. Shape: (B, T, K*F)
            mask (Tensor): Boolean mask. Shape: (B, T, K), True means masked.
        """
        B, T, _ = src.shape
        K = config['data']['n_kps']
        F = 2

        # --- Input Preparation ---
        src = src.reshape(B, T, K, F)
        src_embedded = self.keypoint_embedder(src)
        pos_embedded = self.pos_encoder(src_embedded) # (B, T, K, D)
        
        # Flatten for token processing
        pos_embedded = pos_embedded.reshape(B, T * K, self.d_model)
        mask_flat = mask.reshape(B, T * K) # (B, N), N=T*K

        # --- Encoder Forward Pass (on visible tokens only) ---
        # `~mask_flat` selects the visible tokens (where mask is False)
        visible_tokens = pos_embedded[~mask_flat].reshape(B, -1, self.d_model)
        encoded_tokens = self.transformer_encoder(visible_tokens)

        # --- Decoder Forward Pass ---
        # 1. Project encoded tokens to decoder dimension
        encoded_tokens_decoder = self.encoder_to_decoder(encoded_tokens)

        # 2. Create the full sequence for the decoder
        # Initialize with mask tokens
        decoder_input = self.mask_token.repeat(B, T * K, 1)
        
        # 3. Place the encoded visible tokens back into their original positions
        # We use the inverse of the mask to find the positions
        decoder_input[~mask_flat] = encoded_tokens_decoder.reshape(-1, config['model']['decoder_d_model'])

        # 4. Run the decoder
        decoded_tokens = self.transformer_decoder(decoder_input)

        # --- Output Processing ---
        output = self.output_head(decoded_tokens) # (B, T*K, F)
        output = output.reshape(B, T, K * F) # (B, T, K*F)
        
        return output
