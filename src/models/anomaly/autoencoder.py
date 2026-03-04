"""
Variational Autoencoder for Anomaly-Based Fraud Detection

Unsupervised anomaly detection using VAEs to identify fraudulent transactions
based on reconstruction error.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from loguru import logger
import numpy as np


class VAEEncoder(nn.Module):
    """Encoder network for VAE"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64, 32],
        latent_dim: int = 16
    ):
        super().__init__()
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample from N(mu, sigma^2)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEDecoder(nn.Module):
    """Decoder network for VAE"""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list = [32, 64, 128],
        output_dim: int = None
    ):
        super().__init__()
        
        if output_dim is None:
            output_dim = hidden_dims[0] * 2
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[0], output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to reconstruction"""
        return self.decoder(z)


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for anomaly detection
    
    Learns compressed representation of normal transactions.
    High reconstruction error indicates potential fraud.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: list = [128, 64, 32],
        beta: float = 1.0
    ):
        """
        Args:
            input_dim: Number of input features
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            beta: Weight for KL divergence term in loss
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        self.encoder = VAEEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = VAEDecoder(latent_dim, list(reversed(hidden_dims)), input_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder and decoder
        
        Returns:
            x_recon: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar
    
    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss (reconstruction + KL divergence)
        
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss (BCE or MSE)
            kl_loss: KL divergence loss
        """
        # Reconstruction loss (MSE for continuous features)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def compute_anomaly_scores(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores based on reconstruction error
        
        Returns:
            scores: Anomaly score for each sample (higher = more anomalous)
        """
        self.eval()
        
        with torch.no_grad():
            x_recon, _, _ = self.forward(X)
            
            # Per-sample reconstruction error
            errors = torch.mean((X - x_recon) ** 2, dim=1)
            
            return errors
    
    def detect_anomalies(
        self,
        X: torch.Tensor,
        threshold: Optional[float] = None,
        contamination: float = 0.01
    ) -> torch.Tensor:
        """
        Detect anomalies based on reconstruction error threshold
        
        Args:
            X: Input data
            threshold: Anomaly threshold (if None, use contamination-based)
            contamination: Expected proportion of anomalies
            
        Returns:
            predictions: 1 for anomaly, 0 for normal
        """
        scores = self.compute_anomaly_scores(X)
        
        if threshold is None:
            # Use percentile-based threshold
            threshold = torch.quantile(scores, 1 - contamination)
        
        predictions = (scores > threshold).long()
        
        return predictions, scores


class VAEAnomalyDetector:
    """High-level API for VAE-based anomaly detection"""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: list = [128, 64, 32],
        lr: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model = VariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.threshold = None
        
    def fit(
        self,
        X_train: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.1,
        verbose: bool = True
    ):
        """Train VAE on normal transactions only"""
        
        # Split into train/validation
        n_train = int(len(X_train) * (1 - validation_split))
        X_tr = X_train[:n_train]
        X_val = X_train[n_train:]
        
        train_dataset = torch.utils.data.TensorDataset(X_tr)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_recon = 0
            total_kl = 0
            
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                x_recon, mu, logvar = self.model(batch)
                
                # Compute loss
                total_batch_loss, recon_loss, kl_loss = self.model.compute_loss(
                    batch, x_recon, mu, logvar
                )
                
                # Backward pass
                total_batch_loss.backward()
                self.optimizer.step()
                
                total_loss += total_batch_loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
            
            avg_loss = total_loss / len(X_tr)
            avg_recon = total_recon / len(X_tr)
            avg_kl = total_kl / len(X_tr)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    x_recon_val, mu_val, logvar_val = self.model(X_val.to(self.device))
                    val_loss, _, _ = self.model.compute_loss(
                        X_val.to(self.device), x_recon_val, mu_val, logvar_val
                    )
                    val_loss = val_loss / len(X_val)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                if verbose:
                    logger.info(f"Epoch {epoch+1}/{epochs} - "
                              f"Train Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, "
                              f"KL: {avg_kl:.4f}), Val Loss: {val_loss:.4f}")
            else:
                if verbose:
                    logger.info(f"Epoch {epoch+1}/{epochs} - "
                              f"Train Loss: {avg_loss:.4f}")
        
        # Compute threshold on training data
        self._compute_threshold(X_tr)
    
    def _compute_threshold(self, X: torch.Tensor, percentile: float = 99):
        """Compute anomaly threshold based on training data"""
        self.model.eval()
        
        with torch.no_grad():
            scores = self.model.compute_anomaly_scores(X.to(self.device))
            self.threshold = torch.percentile(scores, percentile).item()
        
        logger.info(f"Anomaly threshold set to: {self.threshold:.4f}")
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict anomalies"""
        if self.threshold is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        with torch.no_grad():
            predictions, scores = self.model.detect_anomalies(
                X.to(self.device),
                threshold=self.threshold
            )
            return predictions.cpu(), scores.cpu()
    
    def anomaly_score(self, X: torch.Tensor) -> torch.Tensor:
        """Get anomaly scores"""
        self.model.eval()
        with torch.no_grad():
            return self.model.compute_anomaly_scores(X.to(self.device)).cpu()
