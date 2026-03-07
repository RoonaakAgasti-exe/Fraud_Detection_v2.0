import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any

class VAE(nn.Module):
    """Variational Autoencoder for Anomaly Detection"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], latent_dim: int = 16):
        super(VAE, self).__init__()
        
        # Encoder
        modules = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            modules.append(nn.Linear(last_dim, h_dim))
            modules.append(nn.BatchNorm1d(h_dim))
            modules.append(nn.ReLU())
            last_dim = h_dim
            
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_var = nn.Linear(last_dim, latent_dim)
        
        # Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])
        
        hidden_dims.reverse()
        last_dim = hidden_dims[0]
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(last_dim, hidden_dims[i+1]))
            modules.append(nn.BatchNorm1d(hidden_dims[i+1]))
            modules.append(nn.ReLU())
            last_dim = hidden_dims[i+1]
            
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Linear(last_dim, input_dim)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var
        
    def loss_function(self, recon_x, x, mu, log_var, beta=1.0) -> Dict[str, torch.Tensor]:
        recons_loss = F.mse_loss(recon_x, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        
        loss = recons_loss + beta * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}
        
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error as anomaly score"""
        self.eval()
        with torch.no_grad():
            reconstruction, mu, log_var = self.forward(x)
            # MSE per sample
            score = torch.mean((x - reconstruction)**2, dim=1)
        return score

class VAEAnomalyDetector:
    """Wrapper for VAE model to handle preprocessing and training"""
    
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        self.model = VAE(
            input_dim=input_dim,
            hidden_dims=config.get('hidden_dims', [128, 64]),
            latent_dim=config.get('latent_dim', 16)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 0.001))
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 256)
        self.beta = config.get('beta', 1.0)
        
    def fit(self, X: torch.Tensor):
        self.model.train()
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        from loguru import logger
        logger.info("Training VAE for anomaly detection...")
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_idx, (data,) in enumerate(dataloader):
                self.optimizer.zero_grad()
                recons, mu, log_var = self.model(data)
                losses = self.model.loss_function(recons, data, mu, log_var, beta=self.beta)
                loss = losses['loss']
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.6f}")
                
    def predict_score(self, X: torch.Tensor) -> torch.Tensor:
        return self.model.compute_anomaly_score(X)
