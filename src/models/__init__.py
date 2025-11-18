"""Social recommendation models implementation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import networkx as nx
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional imports for GNN models
try:
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Data, DataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    # Create dummy classes for when torch_geometric is not available
    class GCNConv:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch_geometric is required for GNN models")
    
    class GATConv:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch_geometric is required for GNN models")
    
    class Data:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch_geometric is required for GNN models")
    
    class DataLoader:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch_geometric is required for GNN models")

from src.data.generator import SocialDataset


class BaseSocialRecommender(ABC):
    """Base class for social recommendation models."""
    
    def __init__(self, name: str) -> None:
        """Initialize base recommender.
        
        Args:
            name: Model name.
        """
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, dataset: SocialDataset) -> None:
        """Fit the model to the dataset.
        
        Args:
            dataset: Social recommendation dataset.
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings for a user.
        
        Args:
            user_id: User ID.
            item_ids: Optional list of item IDs to predict for.
            
        Returns:
            Dictionary mapping item_id to predicted rating.
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of (item_id, score) tuples.
        """
        pass


class SocialCollaborativeFiltering(BaseSocialRecommender):
    """Social collaborative filtering using friends' preferences."""
    
    def __init__(self, social_weight: float = 0.3) -> None:
        """Initialize social collaborative filtering.
        
        Args:
            social_weight: Weight for social influence (0-1).
        """
        super().__init__("SocialCollaborativeFiltering")
        self.social_weight = social_weight
        self.user_item_matrix = None
        self.user_similarity = None
        self.social_network = None
        self.user_ids = None
        self.item_ids = None
    
    def fit(self, dataset: SocialDataset) -> None:
        """Fit the social collaborative filtering model.
        
        Args:
            dataset: Social recommendation dataset.
        """
        # Create user-item matrix
        interactions = dataset.interactions
        self.user_ids = sorted(interactions['user_id'].unique())
        self.item_ids = sorted(interactions['item_id'].unique())
        
        user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        item_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}
        
        # Build rating matrix
        self.user_item_matrix = np.zeros((len(self.user_ids), len(self.item_ids)))
        
        for _, row in interactions.iterrows():
            user_idx = user_to_idx[row['user_id']]
            item_idx = item_to_idx[row['item_id']]
            self.user_item_matrix[user_idx, item_idx] = row['rating']
        
        # Compute user similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        
        # Store social network
        self.social_network = dataset.social_network
        
        self.is_fitted = True
    
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings for a user.
        
        Args:
            user_id: User ID.
            item_ids: Optional list of item IDs to predict for.
            
        Returns:
            Dictionary mapping item_id to predicted rating.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id not in self.user_ids:
            return {}
        
        user_idx = self.user_ids.index(user_id)
        predictions = {}
        
        if item_ids is None:
            item_ids = self.item_ids
        
        for item_id in item_ids:
            if item_id not in self.item_ids:
                continue
            
            item_idx = self.item_ids.index(item_id)
            
            # Collaborative filtering prediction
            cf_pred = self._collaborative_prediction(user_idx, item_idx)
            
            # Social influence prediction
            social_pred = self._social_prediction(user_id, item_id)
            
            # Combine predictions
            final_pred = (1 - self.social_weight) * cf_pred + self.social_weight * social_pred
            predictions[item_id] = final_pred
        
        return predictions
    
    def _collaborative_prediction(self, user_idx: int, item_idx: int) -> float:
        """Compute collaborative filtering prediction."""
        user_ratings = self.user_item_matrix[user_idx]
        if np.sum(user_ratings > 0) == 0:
            return 0.0
        
        # Find similar users who rated this item
        item_ratings = self.user_item_matrix[:, item_idx]
        rated_mask = item_ratings > 0
        
        if np.sum(rated_mask) == 0:
            return 0.0
        
        similarities = self.user_similarity[user_idx][rated_mask]
        ratings = item_ratings[rated_mask]
        
        if np.sum(np.abs(similarities)) == 0:
            return np.mean(ratings)
        
        return np.sum(similarities * ratings) / np.sum(np.abs(similarities))
    
    def _social_prediction(self, user_id: str, item_id: str) -> float:
        """Compute social influence prediction."""
        friends = self.social_network.get(user_id, [])
        if not friends:
            return 0.0
        
        friend_ratings = []
        for friend_id in friends:
            if friend_id in self.user_ids:
                friend_idx = self.user_ids.index(friend_id)
                item_idx = self.item_ids.index(item_id)
                rating = self.user_item_matrix[friend_idx, item_idx]
                if rating > 0:
                    friend_ratings.append(rating)
        
        return np.mean(friend_ratings) if friend_ratings else 0.0
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of (item_id, score) tuples.
        """
        predictions = self.predict(user_id)
        
        # Filter out already rated items
        user_idx = self.user_ids.index(user_id) if user_id in self.user_ids else -1
        if user_idx >= 0:
            rated_items = set()
            for item_idx, rating in enumerate(self.user_item_matrix[user_idx]):
                if rating > 0:
                    rated_items.add(self.item_ids[item_idx])
            
            predictions = {k: v for k, v in predictions.items() if k not in rated_items}
        
        # Sort by score and return top N
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]


class SocialMatrixFactorization(BaseSocialRecommender):
    """Social matrix factorization with social regularization."""
    
    def __init__(self, n_factors: int = 50, reg_param: float = 0.01, 
                 social_reg: float = 0.1, n_iterations: int = 100) -> None:
        """Initialize social matrix factorization.
        
        Args:
            n_factors: Number of latent factors.
            reg_param: Regularization parameter.
            social_reg: Social regularization parameter.
            n_iterations: Number of iterations.
        """
        super().__init__("SocialMatrixFactorization")
        self.n_factors = n_factors
        self.reg_param = reg_param
        self.social_reg = social_reg
        self.n_iterations = n_iterations
        
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        self.user_ids = None
        self.item_ids = None
        self.social_network = None
    
    def fit(self, dataset: SocialDataset) -> None:
        """Fit the social matrix factorization model.
        
        Args:
            dataset: Social recommendation dataset.
        """
        interactions = dataset.interactions
        self.user_ids = sorted(interactions['user_id'].unique())
        self.item_ids = sorted(interactions['item_id'].unique())
        self.social_network = dataset.social_network
        
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = interactions['rating'].mean()
        
        # Create user-item matrix
        user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        item_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}
        
        # Training loop
        for iteration in range(self.n_iterations):
            for _, row in interactions.iterrows():
                user_idx = user_to_idx[row['user_id']]
                item_idx = item_to_idx[row['item_id']]
                rating = row['rating']
                
                # Predict rating
                pred = (self.global_bias + 
                       self.user_bias[user_idx] + 
                       self.item_bias[item_idx] + 
                       np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
                
                error = rating - pred
                
                # Update factors
                user_factor_old = self.user_factors[user_idx].copy()
                self.user_factors[user_idx] += 0.01 * (error * self.item_factors[item_idx] - 
                                                     self.reg_param * self.user_factors[user_idx])
                self.item_factors[item_idx] += 0.01 * (error * user_factor_old - 
                                                     self.reg_param * self.item_factors[item_idx])
                
                # Update biases
                self.user_bias[user_idx] += 0.01 * (error - self.reg_param * self.user_bias[user_idx])
                self.item_bias[item_idx] += 0.01 * (error - self.reg_param * self.item_bias[item_idx])
            
            # Social regularization
            self._social_regularization()
        
        self.is_fitted = True
    
    def _social_regularization(self) -> None:
        """Apply social regularization to user factors."""
        for user_id, friends in self.social_network.items():
            if user_id not in self.user_ids:
                continue
            
            user_idx = self.user_ids.index(user_id)
            if not friends:
                continue
            
            # Compute average friend factors
            friend_factors = []
            for friend_id in friends:
                if friend_id in self.user_ids:
                    friend_idx = self.user_ids.index(friend_id)
                    friend_factors.append(self.user_factors[friend_idx])
            
            if friend_factors:
                avg_friend_factors = np.mean(friend_factors, axis=0)
                # Regularize user factors towards friends
                self.user_factors[user_idx] += self.social_reg * (avg_friend_factors - self.user_factors[user_idx])
    
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings for a user.
        
        Args:
            user_id: User ID.
            item_ids: Optional list of item IDs to predict for.
            
        Returns:
            Dictionary mapping item_id to predicted rating.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id not in self.user_ids:
            return {}
        
        user_idx = self.user_ids.index(user_id)
        predictions = {}
        
        if item_ids is None:
            item_ids = self.item_ids
        
        for item_id in item_ids:
            if item_id not in self.item_ids:
                continue
            
            item_idx = self.item_ids.index(item_id)
            pred = (self.global_bias + 
                   self.user_bias[user_idx] + 
                   self.item_bias[item_idx] + 
                   np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
            
            predictions[item_id] = max(1.0, min(5.0, pred))  # Clip to rating range
        
        return predictions
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of (item_id, score) tuples.
        """
        predictions = self.predict(user_id)
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]


class SocialGraphNeuralNetwork(BaseSocialRecommender):
    """Social recommendation using Graph Neural Networks."""
    
    def __init__(self, hidden_dim: int = 64, n_layers: int = 2, dropout: float = 0.1) -> None:
        """Initialize social GNN model.
        
        Args:
            hidden_dim: Hidden dimension size.
            n_layers: Number of GNN layers.
            dropout: Dropout rate.
        """
        super().__init__("SocialGraphNeuralNetwork")
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.model = None
        self.user_ids = None
        self.item_ids = None
        self.social_network = None
    
    def fit(self, dataset: SocialDataset) -> None:
        """Fit the social GNN model.
        
        Args:
            dataset: Social recommendation dataset.
        """
        self.user_ids = sorted(dataset.interactions['user_id'].unique())
        self.item_ids = sorted(dataset.interactions['item_id'].unique())
        self.social_network = dataset.social_network
        
        # Create graph data
        graph_data = self._create_graph_data(dataset)
        
        # Initialize model
        self.model = SocialGNNModel(
            num_users=len(self.user_ids),
            num_items=len(self.item_ids),
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout
        )
        
        # Training would go here (simplified for demo)
        self.is_fitted = True
    
    def _create_graph_data(self, dataset: SocialDataset) -> Data:
        """Create PyTorch Geometric graph data."""
        # This is a simplified implementation
        # In practice, you'd create proper edge indices for user-item and social relationships
        num_nodes = len(self.user_ids) + len(self.item_ids)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return Data(edge_index=edge_index, num_nodes=num_nodes)
    
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings for a user.
        
        Args:
            user_id: User ID.
            item_ids: Optional list of item IDs to predict for.
            
        Returns:
            Dictionary mapping item_id to predicted rating.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Simplified prediction (in practice, you'd use the trained GNN)
        predictions = {}
        if item_ids is None:
            item_ids = self.item_ids
        
        for item_id in item_ids:
            # Placeholder prediction
            predictions[item_id] = np.random.uniform(1, 5)
        
        return predictions
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of (item_id, score) tuples.
        """
        predictions = self.predict(user_id)
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]


class SocialGNNModel(nn.Module):
    """Graph Neural Network model for social recommendation."""
    
    def __init__(self, num_users: int, num_items: int, hidden_dim: int, 
                 n_layers: int, dropout: float) -> None:
        """Initialize GNN model.
        
        Args:
            num_users: Number of users.
            num_items: Number of items.
            hidden_dim: Hidden dimension size.
            n_layers: Number of GNN layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Prediction layer
        self.predict_layer = nn.Linear(hidden_dim * 2, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            user_ids: User ID tensor.
            item_ids: Item ID tensor.
            edge_index: Graph edge indices.
            
        Returns:
            Predicted ratings.
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Combine user and item embeddings
        combined_emb = torch.cat([user_emb, item_emb], dim=1)
        
        # Apply GNN layers
        x = combined_emb
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Predict ratings
        ratings = self.predict_layer(x)
        return ratings.squeeze()


class PopularityBaseline(BaseSocialRecommender):
    """Popularity-based baseline recommender."""
    
    def __init__(self) -> None:
        """Initialize popularity baseline."""
        super().__init__("PopularityBaseline")
        self.item_popularity = None
        self.item_ids = None
    
    def fit(self, dataset: SocialDataset) -> None:
        """Fit the popularity baseline.
        
        Args:
            dataset: Social recommendation dataset.
        """
        interactions = dataset.interactions
        self.item_ids = sorted(interactions['item_id'].unique())
        
        # Compute item popularity (average rating)
        self.item_popularity = interactions.groupby('item_id')['rating'].mean().to_dict()
        
        self.is_fitted = True
    
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings for a user.
        
        Args:
            user_id: User ID.
            item_ids: Optional list of item IDs to predict for.
            
        Returns:
            Dictionary mapping item_id to predicted rating.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if item_ids is None:
            item_ids = self.item_ids
        
        predictions = {}
        for item_id in item_ids:
            predictions[item_id] = self.item_popularity.get(item_id, 0.0)
        
        return predictions
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of (item_id, score) tuples.
        """
        predictions = self.predict(user_id)
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]
    
    def __init__(self, name: str) -> None:
        """Initialize base recommender.
        
        Args:
            name: Model name.
        """
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, dataset: SocialDataset) -> None:
        """Fit the model to the dataset.
        
        Args:
            dataset: Social recommendation dataset.
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings for a user.
        
        Args:
            user_id: User ID.
            item_ids: Optional list of item IDs to predict for.
            
        Returns:
            Dictionary mapping item_id to predicted rating.
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of (item_id, score) tuples.
        """
        pass


class SocialCollaborativeFiltering(BaseSocialRecommender):
    """Social collaborative filtering using friends' preferences."""
    
    def __init__(self, social_weight: float = 0.3) -> None:
        """Initialize social collaborative filtering.
        
        Args:
            social_weight: Weight for social influence (0-1).
        """
        super().__init__("SocialCollaborativeFiltering")
        self.social_weight = social_weight
        self.user_item_matrix = None
        self.user_similarity = None
        self.social_network = None
        self.user_ids = None
        self.item_ids = None
    
    def fit(self, dataset: SocialDataset) -> None:
        """Fit the social collaborative filtering model.
        
        Args:
            dataset: Social recommendation dataset.
        """
        # Create user-item matrix
        interactions = dataset.interactions
        self.user_ids = sorted(interactions['user_id'].unique())
        self.item_ids = sorted(interactions['item_id'].unique())
        
        user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        item_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}
        
        # Build rating matrix
        self.user_item_matrix = np.zeros((len(self.user_ids), len(self.item_ids)))
        
        for _, row in interactions.iterrows():
            user_idx = user_to_idx[row['user_id']]
            item_idx = item_to_idx[row['item_id']]
            self.user_item_matrix[user_idx, item_idx] = row['rating']
        
        # Compute user similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        
        # Store social network
        self.social_network = dataset.social_network
        
        self.is_fitted = True
    
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings for a user.
        
        Args:
            user_id: User ID.
            item_ids: Optional list of item IDs to predict for.
            
        Returns:
            Dictionary mapping item_id to predicted rating.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id not in self.user_ids:
            return {}
        
        user_idx = self.user_ids.index(user_id)
        predictions = {}
        
        if item_ids is None:
            item_ids = self.item_ids
        
        for item_id in item_ids:
            if item_id not in self.item_ids:
                continue
            
            item_idx = self.item_ids.index(item_id)
            
            # Collaborative filtering prediction
            cf_pred = self._collaborative_prediction(user_idx, item_idx)
            
            # Social influence prediction
            social_pred = self._social_prediction(user_id, item_id)
            
            # Combine predictions
            final_pred = (1 - self.social_weight) * cf_pred + self.social_weight * social_pred
            predictions[item_id] = final_pred
        
        return predictions
    
    def _collaborative_prediction(self, user_idx: int, item_idx: int) -> float:
        """Compute collaborative filtering prediction."""
        user_ratings = self.user_item_matrix[user_idx]
        if np.sum(user_ratings > 0) == 0:
            return 0.0
        
        # Find similar users who rated this item
        item_ratings = self.user_item_matrix[:, item_idx]
        rated_mask = item_ratings > 0
        
        if np.sum(rated_mask) == 0:
            return 0.0
        
        similarities = self.user_similarity[user_idx][rated_mask]
        ratings = item_ratings[rated_mask]
        
        if np.sum(np.abs(similarities)) == 0:
            return np.mean(ratings)
        
        return np.sum(similarities * ratings) / np.sum(np.abs(similarities))
    
    def _social_prediction(self, user_id: str, item_id: str) -> float:
        """Compute social influence prediction."""
        friends = self.social_network.get(user_id, [])
        if not friends:
            return 0.0
        
        friend_ratings = []
        for friend_id in friends:
            if friend_id in self.user_ids:
                friend_idx = self.user_ids.index(friend_id)
                item_idx = self.item_ids.index(item_id)
                rating = self.user_item_matrix[friend_idx, item_idx]
                if rating > 0:
                    friend_ratings.append(rating)
        
        return np.mean(friend_ratings) if friend_ratings else 0.0
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of (item_id, score) tuples.
        """
        predictions = self.predict(user_id)
        
        # Filter out already rated items
        user_idx = self.user_ids.index(user_id) if user_id in self.user_ids else -1
        if user_idx >= 0:
            rated_items = set()
            for item_idx, rating in enumerate(self.user_item_matrix[user_idx]):
                if rating > 0:
                    rated_items.add(self.item_ids[item_idx])
            
            predictions = {k: v for k, v in predictions.items() if k not in rated_items}
        
        # Sort by score and return top N
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]


class SocialMatrixFactorization(BaseSocialRecommender):
    """Social matrix factorization with social regularization."""
    
    def __init__(self, n_factors: int = 50, reg_param: float = 0.01, 
                 social_reg: float = 0.1, n_iterations: int = 100) -> None:
        """Initialize social matrix factorization.
        
        Args:
            n_factors: Number of latent factors.
            reg_param: Regularization parameter.
            social_reg: Social regularization parameter.
            n_iterations: Number of iterations.
        """
        super().__init__("SocialMatrixFactorization")
        self.n_factors = n_factors
        self.reg_param = reg_param
        self.social_reg = social_reg
        self.n_iterations = n_iterations
        
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        self.user_ids = None
        self.item_ids = None
        self.social_network = None
    
    def fit(self, dataset: SocialDataset) -> None:
        """Fit the social matrix factorization model.
        
        Args:
            dataset: Social recommendation dataset.
        """
        interactions = dataset.interactions
        self.user_ids = sorted(interactions['user_id'].unique())
        self.item_ids = sorted(interactions['item_id'].unique())
        self.social_network = dataset.social_network
        
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = interactions['rating'].mean()
        
        # Create user-item matrix
        user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        item_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}
        
        # Training loop
        for iteration in range(self.n_iterations):
            for _, row in interactions.iterrows():
                user_idx = user_to_idx[row['user_id']]
                item_idx = item_to_idx[row['item_id']]
                rating = row['rating']
                
                # Predict rating
                pred = (self.global_bias + 
                       self.user_bias[user_idx] + 
                       self.item_bias[item_idx] + 
                       np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
                
                error = rating - pred
                
                # Update factors
                user_factor_old = self.user_factors[user_idx].copy()
                self.user_factors[user_idx] += 0.01 * (error * self.item_factors[item_idx] - 
                                                     self.reg_param * self.user_factors[user_idx])
                self.item_factors[item_idx] += 0.01 * (error * user_factor_old - 
                                                     self.reg_param * self.item_factors[item_idx])
                
                # Update biases
                self.user_bias[user_idx] += 0.01 * (error - self.reg_param * self.user_bias[user_idx])
                self.item_bias[item_idx] += 0.01 * (error - self.reg_param * self.item_bias[item_idx])
            
            # Social regularization
            self._social_regularization()
        
        self.is_fitted = True
    
    def _social_regularization(self) -> None:
        """Apply social regularization to user factors."""
        for user_id, friends in self.social_network.items():
            if user_id not in self.user_ids:
                continue
            
            user_idx = self.user_ids.index(user_id)
            if not friends:
                continue
            
            # Compute average friend factors
            friend_factors = []
            for friend_id in friends:
                if friend_id in self.user_ids:
                    friend_idx = self.user_ids.index(friend_id)
                    friend_factors.append(self.user_factors[friend_idx])
            
            if friend_factors:
                avg_friend_factors = np.mean(friend_factors, axis=0)
                # Regularize user factors towards friends
                self.user_factors[user_idx] += self.social_reg * (avg_friend_factors - self.user_factors[user_idx])
    
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings for a user.
        
        Args:
            user_id: User ID.
            item_ids: Optional list of item IDs to predict for.
            
        Returns:
            Dictionary mapping item_id to predicted rating.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id not in self.user_ids:
            return {}
        
        user_idx = self.user_ids.index(user_id)
        predictions = {}
        
        if item_ids is None:
            item_ids = self.item_ids
        
        for item_id in item_ids:
            if item_id not in self.item_ids:
                continue
            
            item_idx = self.item_ids.index(item_id)
            pred = (self.global_bias + 
                   self.user_bias[user_idx] + 
                   self.item_bias[item_idx] + 
                   np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
            
            predictions[item_id] = max(1.0, min(5.0, pred))  # Clip to rating range
        
        return predictions
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of (item_id, score) tuples.
        """
        predictions = self.predict(user_id)
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]


class SocialGraphNeuralNetwork(BaseSocialRecommender):
    """Social recommendation using Graph Neural Networks."""
    
    def __init__(self, hidden_dim: int = 64, n_layers: int = 2, dropout: float = 0.1) -> None:
        """Initialize social GNN model.
        
        Args:
            hidden_dim: Hidden dimension size.
            n_layers: Number of GNN layers.
            dropout: Dropout rate.
        """
        super().__init__("SocialGraphNeuralNetwork")
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.model = None
        self.user_ids = None
        self.item_ids = None
        self.social_network = None
    
    def fit(self, dataset: SocialDataset) -> None:
        """Fit the social GNN model.
        
        Args:
            dataset: Social recommendation dataset.
        """
        self.user_ids = sorted(dataset.interactions['user_id'].unique())
        self.item_ids = sorted(dataset.interactions['item_id'].unique())
        self.social_network = dataset.social_network
        
        # Create graph data
        graph_data = self._create_graph_data(dataset)
        
        # Initialize model
        self.model = SocialGNNModel(
            num_users=len(self.user_ids),
            num_items=len(self.item_ids),
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout
        )
        
        # Training would go here (simplified for demo)
        self.is_fitted = True
    
    def _create_graph_data(self, dataset: SocialDataset) -> Data:
        """Create PyTorch Geometric graph data."""
        # This is a simplified implementation
        # In practice, you'd create proper edge indices for user-item and social relationships
        num_nodes = len(self.user_ids) + len(self.item_ids)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return Data(edge_index=edge_index, num_nodes=num_nodes)
    
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings for a user.
        
        Args:
            user_id: User ID.
            item_ids: Optional list of item IDs to predict for.
            
        Returns:
            Dictionary mapping item_id to predicted rating.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Simplified prediction (in practice, you'd use the trained GNN)
        predictions = {}
        if item_ids is None:
            item_ids = self.item_ids
        
        for item_id in item_ids:
            # Placeholder prediction
            predictions[item_id] = np.random.uniform(1, 5)
        
        return predictions
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of (item_id, score) tuples.
        """
        predictions = self.predict(user_id)
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]


class SocialGNNModel(nn.Module):
    """Graph Neural Network model for social recommendation."""
    
    def __init__(self, num_users: int, num_items: int, hidden_dim: int, 
                 n_layers: int, dropout: float) -> None:
        """Initialize GNN model.
        
        Args:
            num_users: Number of users.
            num_items: Number of items.
            hidden_dim: Hidden dimension size.
            n_layers: Number of GNN layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Prediction layer
        self.predict_layer = nn.Linear(hidden_dim * 2, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            user_ids: User ID tensor.
            item_ids: Item ID tensor.
            edge_index: Graph edge indices.
            
        Returns:
            Predicted ratings.
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Combine user and item embeddings
        combined_emb = torch.cat([user_emb, item_emb], dim=1)
        
        # Apply GNN layers
        x = combined_emb
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Predict ratings
        ratings = self.predict_layer(x)
        return ratings.squeeze()


class PopularityBaseline(BaseSocialRecommender):
    """Popularity-based baseline recommender."""
    
    def __init__(self) -> None:
        """Initialize popularity baseline."""
        super().__init__("PopularityBaseline")
        self.item_popularity = None
        self.item_ids = None
    
    def fit(self, dataset: SocialDataset) -> None:
        """Fit the popularity baseline.
        
        Args:
            dataset: Social recommendation dataset.
        """
        interactions = dataset.interactions
        self.item_ids = sorted(interactions['item_id'].unique())
        
        # Compute item popularity (average rating)
        self.item_popularity = interactions.groupby('item_id')['rating'].mean().to_dict()
        
        self.is_fitted = True
    
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings for a user.
        
        Args:
            user_id: User ID.
            item_ids: Optional list of item IDs to predict for.
            
        Returns:
            Dictionary mapping item_id to predicted rating.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if item_ids is None:
            item_ids = self.item_ids
        
        predictions = {}
        for item_id in item_ids:
            predictions[item_id] = self.item_popularity.get(item_id, 0.0)
        
        return predictions
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of (item_id, score) tuples.
        """
        predictions = self.predict(user_id)
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]
