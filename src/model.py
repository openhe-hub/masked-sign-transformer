import torch
import torch.nn as nn
import math

from config_loader import config

class PositionalEncoding(nn.Module):
    # This class remains unchanged, expects (seq_len, batch, dim)
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PoseTransformer(nn.Module):
    def __init__(self):
        super(PoseTransformer, self).__init__()
        
        n_kps = config['data']['n_kps']
        features_per_kp = config['data']['features_per_kp']
        d_model = config['model']['d_model']
        n_head = config['model']['n_head']
        num_encoder_layers = config['model']['num_encoder_layers']
        dim_feedforward = config['model']['dim_feedforward']
        dropout = config['model']['dropout']
        
        self.model_features_per_kp =2
        input_features = n_kps * self.model_features_per_kp
        
        self.input_projection = nn.Linear(input_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Sticking with batch_first=False as PositionalEncoding expects (seq, batch, feature)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.output_head = nn.Linear(d_model, input_features)
        self.d_model = d_model

        # Add a learnable mask token. Note its shape matches the input features, not d_model.
        self.mask_token = nn.Parameter(torch.randn(1, 1, input_features))

    def forward(self, src, mask):
        """
        Args:
            src (Tensor): The masked input sequence from the dataloader.
                          Shape: (batch_size, sequence_length, input_features)
            mask (Tensor): The boolean mask indicating which keypoints were masked.
                           Shape: (batch_size, sequence_length, n_kps)
        """
        
        # 1. Replace masked parts of the input with the learnable mask_token
        n_kps = config['data']['n_kps']
        # features_per_kp = config['data']['features_per_kp']
        
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, -1, self.model_features_per_kp)
        expanded_mask = expanded_mask.reshape(src.shape[0], src.shape[1], -1)

        src = torch.where(expanded_mask, self.mask_token.expand_as(src), src)

        # 2. Permute src from (batch, seq, feature) to (seq, batch, feature) for the encoder
        src = src.permute(1, 0, 2)

        # 3. Project, add positional encoding, and pass through transformer
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # No causal mask is needed for MAE
        output = self.transformer_encoder(src, mask=None)
        
        # 4. Project back to the original feature space
        output = self.output_head(output)
        
        # 5. Permute output back to (batch, seq, feature)
        output = output.permute(1, 0, 2)
        
        return output