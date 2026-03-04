"""
TabNet Implementation for Tabular Fraud Detection

PyTorch implementation of TabNet architecture for interpretable
tabular data classification with attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from loguru import logger


class GLU(nn.Module):
    """Gated Linear Unit"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return F.glu(x, dim=-1)


class FeatureTransformer(nn.Module):
    """Feature transformation block with GLU"""
    
    def __init__(self, input_dim: int, output_dim: int, n_shared: int = 2, n_independent: int = 2):
        super().__init__()
        
        # Shared layers across decision steps
        self.shared_layers = nn.ModuleList()
        for _ in range(n_shared):
            self.shared_layers.append(GLU(input_dim, output_dim))
            input_dim = output_dim
        
        # Independent layers
        self.independent_layers = nn.ModuleList()
        for _ in range(n_independent):
            self.independent_layers.append(GLU(output_dim, output_dim))
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply shared layers
        for layer in self.shared_layers:
            x = layer(x)
        
        # Apply independent layers
        for layer in self.independent_layers:
            x = layer(x)
        
        x = self.bn(x)
        return x


class AttentiveTransformer(nn.Module):
    """Attention mechanism for feature selection"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        
    def forward(self, hidden: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: Hidden state from feature transformer
            prior: Prior scaling from previous step
        """
        x = self.fc(hidden)
        x = self.bn(x)
        x = x * prior
        return F.softmax(x, dim=-1)


class TabNetBlock(nn.Module):
    """Single TabNet decision step"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_d: int,
        n_a: int,
        momentum: float = 0.02
    ):
        super().__init__()
        
        self.n_d = n_d  # Decision dimension
        self.n_a = n_a  # Attention dimension
        
        # Feature transformer for attention
        self.attention_transformer = AttentiveTransformer(input_dim, n_a)
        
        # Feature transformer for decision
        self.feature_transformer = FeatureTransformer(n_a, n_d)
        
        # Residual connection
        if input_dim != n_d:
            self.residual_fc = nn.Linear(input_dim, n_d)
        else:
            self.residual_fc = None
            
        # Batch norm
        self.bn = nn.BatchNorm1d(n_d, momentum=momentum)
        
    def forward(
        self,
        x: torch.Tensor,
        prior: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Compute attention weights
        attention_weights = self.attention_transformer(x, prior)
        
        # Apply attention to input
        attended_x = x * attention_weights
        
        # Transform features
        out = self.feature_transformer(attended_x)
        
        # Residual connection
        if self.residual_fc is not None:
            residual = self.residual_fc(x)
        else:
            residual = x[:, :self.n_d]
        
        out = out + residual
        out = self.bn(out)
        out = F.relu(out)
        
        # Update prior for next step
        new_prior = prior - attention_weights
        
        return out, attention_weights, new_prior


class TabNet(nn.Module):
    """
    TabNet: Attentive Interpretable Tabular Learning
    
    Architecture that combines sequential attention with feature transformation
    for interpretable tabular data classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        momentum: float = 0.02,
        dropout_rate: float = 0.1,
        lambda_sparse: float = 0.001
    ):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes (1 for binary)
            n_d: Dimension of decision space
            n_a: Dimension of attention space
            n_steps: Number of decision steps
            gamma: Coefficient for forgetting in attention mechanism
            momentum: Momentum for batch normalization
            dropout_rate: Dropout rate
            lambda_sparse: Sparsity regularization coefficient
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        
        # Initial embedding
        self.embedding = nn.Linear(input_dim, n_d)
        self.bn_init = nn.BatchNorm1d(n_d, momentum=momentum)
        
        # Decision steps
        self.tabnet_blocks = nn.ModuleList([
            TabNetBlock(n_d, output_dim, n_d, n_a, momentum)
            for _ in range(n_steps)
        ])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(n_d, n_d // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_d // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            predictions: Model output logits
            masks: Attention masks for interpretability
        """
        batch_size = x.size(0)
        
        # Initialize
        out = self.embedding(x)
        out = self.bn_init(out)
        out = F.relu(out)
        
        # Prior for attention (all features equally important initially)
        prior = torch.ones((batch_size, self.input_dim), device=x.device)
        
        # Accumulate outputs and masks
        accumulated_output = torch.zeros((batch_size, self.n_d), device=x.device)
        masks = []
        
        # Pass through decision steps
        for step_idx, block in enumerate(self.tabnet_blocks):
            out_step, attention_mask, prior = block(out, prior)
            
            # Apply ReLU and accumulate
            out_step = F.relu(out_step)
            accumulated_output += out_step
            
            # Store mask for interpretability
            masks.append(attention_mask)
            
            # Update prior with forgetting factor
            prior = prior * self.gamma
        
        # Final prediction
        accumulated_output = self.dropout(accumulated_output)
        predictions = self.classifier(accumulated_output)
        
        # Stack masks for visualization
        masks = torch.stack(masks, dim=1)  # (batch, n_steps, input_dim)
        
        return predictions, masks
    
    def compute_sparse_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """Compute sparsity regularization loss"""
        # Encourage sparse feature selection
        entropy = -torch.sum(masks * torch.log(masks + 1e-6), dim=-1)
        mean_entropy = torch.mean(entropy)
        return self.lambda_sparse * mean_entropy
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            return torch.sigmoid(logits)
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature importance from attention masks"""
        self.eval()
        with torch.no_grad():
            _, masks = self.forward(x)
            # Average attention across steps
            importance = masks.mean(dim=1)
            return importance


class TabNetClassifier:
    """High-level API for TabNet training"""
    
    def __init__(
        self,
        input_dim: int,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        lr: float = 0.02,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model = TabNet(
            input_dim=input_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        epochs: int = 50,
        batch_size: int = 1024,
        verbose: bool = True
    ):
        """Train the model"""
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).float()
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions, masks = self.model(batch_X)
                
                # Compute losses
                task_loss = self.criterion(predictions, batch_y.unsqueeze(-1))
                sparse_loss = self.model.compute_sparse_loss(masks)
                total_batch_loss = task_loss + sparse_loss
                
                # Backward pass
                total_batch_loss.backward()
                self.optimizer.step()
                
                total_loss += total_batch_loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds, _ = self.model(X_val.to(self.device))
                    val_loss = self.criterion(
                        val_preds,
                        y_val.to(self.device).unsqueeze(-1).float()
                    ).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                if verbose:
                    logger.info(f"Epoch {epoch+1}/{epochs} - "
                              f"Train Loss: {avg_loss:.4f}, "
                              f"Val Loss: {val_loss:.4f}")
            else:
                if verbose:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            probs = self.model.predict_proba(X.to(self.device))
            return (probs > 0.5).long()
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities"""
        self.model.eval()
        with torch.no_grad():
            return self.model.predict_proba(X.to(self.device))
