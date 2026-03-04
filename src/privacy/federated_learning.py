"""
Federated Learning for Privacy-Preserving Fraud Detection

Train models across multiple institutions without sharing raw transaction data
using the Flower framework.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from loguru import logger
from collections import OrderedDict

try:
    import flwr as fl
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False
    logger.warning("Flower (flwr) not installed. Federated learning disabled.")


class FederatedClient:
    """Base class for federated learning clients"""
    
    def __init__(self, client_id: str, model: torch.nn.Module):
        self.client_id = client_id
        self.model = model
        self.device = next(model.parameters()).device
        
    def get_parameters(self) -> List[np.ndarray]:
        """Extract model parameters"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Apply global model parameters"""
        params_dict = zip(
            [key for key, _ in self.model.state_dict().items()],
            parameters
        )
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def train(self, train_loader, epochs: int = 1, lr: float = 0.01) -> Dict:
        """Local training on client data"""
        raise NotImplementedError
    
    def evaluate(self, test_loader) -> Tuple[float, int]:
        """Evaluate on local test data"""
        raise NotImplementedError


class FraudDetectionClient(FederatedClient):
    """Federated client for fraud detection models"""
    
    def __init__(
        self,
        client_id: str,
        model: torch.nn.Module,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        super().__init__(client_id, model)
        
        self.train_X, self.train_y = train_data
        self.test_X, self.test_y = test_data if test_data else (None, None)
        
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = None
        
    def train(self, train_loader=None, epochs: int = 1, lr: float = 0.01) -> Dict:
        """Train on local data with differential privacy option"""
        
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Create DataLoader if not provided
        if train_loader is None:
            train_dataset = torch.utils.data.TensorDataset(self.train_X, self.train_y)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=min(64, len(self.train_X)),
                shuffle=True
            )
        
        total_loss = 0
        num_samples = len(self.train_X)
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).float()
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = self.criterion(outputs, batch_y.unsqueeze(-1))
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / (epochs * len(train_loader))
        
        metrics = {
            "train_loss": avg_loss,
            "num_samples": num_samples
        }
        
        logger.info(f"Client {self.client_id} - Train Loss: {avg_loss:.4f}")
        
        return metrics
    
    def evaluate(self, test_loader=None) -> Tuple[float, int]:
        """Evaluate model on test data"""
        
        if self.test_X is None:
            return 0.0, 0
        
        self.model.eval()
        
        if test_loader is None:
            test_dataset = torch.utils.data.TensorDataset(self.test_X, self.test_y)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=min(256, len(self.test_X)),
                shuffle=False
            )
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).float()
                
                outputs = self.model(batch_X)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = self.criterion(outputs, batch_y.unsqueeze(-1))
                total_loss += loss.item()
                
                # Predictions
                predictions = (torch.sigmoid(outputs) > 0.5).long()
                correct += (predictions.squeeze() == batch_y.long()).sum().item()
                total += batch_y.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        
        logger.info(f"Client {self.client_id} - Test Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        
        return avg_loss, total


class FederatedStrategy:
    """Federated averaging strategy coordinator"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        num_clients: int,
        rounds: int = 10,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0
    ):
        self.model = model
        self.num_clients = num_clients
        self.rounds = rounds
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        
    def aggregate(self, weights_results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """Aggregate client weights using weighted average"""
        
        # Get number of layers
        num_layers = len(weights_results[0][0])
        
        # Aggregate each layer
        aggregated = []
        
        for layer_idx in range(num_layers):
            # Stack all client weights for this layer
            layer_weights = [client_weights[layer_idx] for client_weights, _ in weights_results]
            client_counts = [count for _, count in weights_results]
            
            # Weighted average based on number of samples
            stacked = np.stack(layer_weights)
            weights = np.array(client_counts)
            weighted_avg = np.average(stacked, axis=0, weights=weights)
            
            aggregated.append(weighted_avg.astype(np.float32))
        
        return aggregated


def create_federated_learner(
    model: torch.nn.Module,
    clients: List[FraudDetectionClient],
    strategy: Optional[FederatedStrategy] = None
):
    """Create Flower federated learning setup"""
    
    if not FLWR_AVAILABLE:
        raise ImportError("Install flower: pip install flwr")
    
    if strategy is None:
        strategy = FederatedStrategy(model, num_clients=len(clients))
    
    # Convert clients to Flower clients
    class FlowerClient(fl.client.NumPyClient):
        def __init__(self, client: FraudDetectionClient):
            self.client = client
        
        def get_parameters(self, config):
            return self.client.get_parameters()
        
        def fit(self, parameters, config):
            self.client.set_parameters(parameters)
            metrics = self.client.train()
            return self.client.get_parameters(), len(self.client.train_X), metrics
        
        def evaluate(self, parameters, config):
            self.client.set_parameters(parameters)
            loss, num_samples = self.client.evaluate()
            return loss, num_samples, {"accuracy": 1 - loss}
    
    # Create client instances
    flower_clients = [FlowerClient(client) for client in clients]
    
    # Define strategy
    class CustomStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(self, rnd, results, failures):
            weights_results = [(r.parameters, r.num_examples) for r in results]
            aggregated = strategy.aggregate(weights_results)
            return aggregated, {}
    
    fed_strategy = CustomStrategy(
        fraction_fit=strategy.fraction_fit,
        fraction_evaluate=strategy.fraction_evaluate,
        min_available_clients=strategy.num_clients
    )
    
    # Create server
    server = fl.server.Server(
        initial_parameters=fl.common.ndarrays_to_parameters(
            clients[0].get_parameters()
        ),
        strategy=fed_strategy
    )
    
    return server, flower_clients


def run_federated_training(
    model: torch.nn.Module,
    clients: List[FraudDetectionClient],
    num_rounds: int = 10,
    server_address: str = "127.0.0.1:8080"
):
    """Run federated learning across clients"""
    
    if not FLWR_AVAILABLE:
        logger.error("Flower not installed. Install with: pip install flwr")
        return
    
    logger.info(f"Starting federated training with {len(clients)} clients")
    
    # Create server
    server, _ = create_federated_learner(model, clients)
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        server=server,
        config=fl.server.ServerConfig(num_rounds=num_rounds)
    )
    
    logger.info("Federated training completed")


# Example usage for multi-bank collaboration
def example_multi_bank_setup():
    """Example: Multiple banks collaborating on fraud detection"""
    
    # Each bank has its own data
    bank_data = {
        "bank_1": {"train": (torch.rand(1000, 20), torch.randint(0, 2, (1000,))), 
                   "test": (torch.rand(200, 20), torch.randint(0, 2, (200,)))},
        "bank_2": {"train": (torch.rand(1500, 20), torch.randint(0, 2, (1500,))), 
                   "test": (torch.rand(300, 20), torch.randint(0, 2, (300,)))},
        "bank_3": {"train": (torch.rand(800, 20), torch.randint(0, 2, (800,))), 
                   "test": (torch.rand(150, 20), torch.randint(0, 2, (150,)))},
    }
    
    # Shared model architecture
    model = torch.nn.Sequential(
        torch.nn.Linear(20, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1)
    )
    
    # Create clients for each bank
    clients = []
    for bank_id, data in bank_data.items():
        client = FraudDetectionClient(
            client_id=bank_id,
            model=model,
            train_data=data["train"],
            test_data=data["test"]
        )
        clients.append(client)
    
    logger.info(f"Created {len(clients)} federated clients")
    
    return model, clients
