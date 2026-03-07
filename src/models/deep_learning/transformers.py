import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional

class PositionalEncoding(nn.Module):
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

class FraudTransformer(nn.Module):
    """Transformer for sequential transaction patterns"""
    
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super(FraudTransformer, self).__init__()
        
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Input embedding (linear projection to d_model)
        self.encoder_input = nn.Linear(input_dim, d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # src shape: [seq_len, batch_size, input_dim]
        
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            self.src_mask = mask

        src = self.encoder_input(src) * math.sqrt(src.size(-1))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        
        # Use the output of the last time step for classification
        output = self.decoder(output[-1])
        return self.sigmoid(output)

class TransformerClassifier:
    """Wrapper for FraudTransformer to handle data batching and training"""
    
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FraudTransformer(
            input_dim=input_dim,
            d_model=config.get('d_model', 64),
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 2),
            dim_feedforward=config.get('dim_feedforward', 128),
            dropout=config.get('dropout', 0.1)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('lr', 0.001))
        self.criterion = nn.BCELoss()
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 32)
        
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        # X shape: [num_samples, seq_len, input_dim]
        # y shape: [num_samples]
        
        # Convert to [seq_len, num_samples, input_dim] for Transformer
        X = X.transpose(0, 1).to(self.device)
        y = y.to(self.device)
        
        self.model.train()
        from loguru import logger
        logger.info("Training FraudTransformer on sequential data...")
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            output = self.model(X).squeeze()
            loss = self.criterion(output, y.float())
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")
                
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            X = X.transpose(0, 1).to(self.device)
            return self.model(X).cpu()
