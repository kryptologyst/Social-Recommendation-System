"""Test script for social recommendation system."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.generator import SocialDataGenerator, SocialDataset
from models.models import (
    SocialCollaborativeFiltering,
    SocialMatrixFactorization,
    PopularityBaseline
)
from utils.evaluation import RecommendationEvaluator


class TestSocialDataGenerator:
    """Test cases for SocialDataGenerator."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = SocialDataGenerator(seed=42)
        assert generator.seed == 42
    
    def test_generate_social_network(self):
        """Test social network generation."""
        generator = SocialDataGenerator(seed=42)
        network = generator.generate_social_network(n_users=10, avg_friends=3)
        
        assert len(network) == 10
        assert all(isinstance(friends, list) for friends in network.values())
    
    def test_generate_items(self):
        """Test item generation."""
        generator = SocialDataGenerator(seed=42)
        items = generator.generate_items(n_items=5)
        
        assert len(items) == 5
        assert 'item_id' in items.columns
        assert 'title' in items.columns
        assert 'category' in items.columns
    
    def test_generate_users(self):
        """Test user generation."""
        generator = SocialDataGenerator(seed=42)
        users = generator.generate_users(n_users=5)
        
        assert len(users) == 5
        assert 'user_id' in users.columns
        assert 'age' in users.columns
    
    def test_generate_dataset(self):
        """Test complete dataset generation."""
        generator = SocialDataGenerator(seed=42)
        dataset = generator.generate_dataset(n_users=50, n_items=25)
        
        assert isinstance(dataset, SocialDataset)
        assert len(dataset.users) == 50
        assert len(dataset.items) == 25
        assert len(dataset.interactions) > 0
        assert len(dataset.social_network) == 50


class TestSocialModels:
    """Test cases for social recommendation models."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        generator = SocialDataGenerator(seed=42)
        return generator.generate_dataset(n_users=20, n_items=10)
    
    def test_social_collaborative_filtering(self, sample_dataset):
        """Test social collaborative filtering model."""
        model = SocialCollaborativeFiltering(social_weight=0.3)
        
        # Test fitting
        model.fit(sample_dataset)
        assert model.is_fitted
        
        # Test prediction
        user_id = sample_dataset.interactions['user_id'].iloc[0]
        predictions = model.predict(user_id)
        assert isinstance(predictions, dict)
        
        # Test recommendation
        recommendations = model.recommend(user_id, 5)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
    
    def test_social_matrix_factorization(self, sample_dataset):
        """Test social matrix factorization model."""
        model = SocialMatrixFactorization(n_factors=10, n_iterations=5)
        
        # Test fitting
        model.fit(sample_dataset)
        assert model.is_fitted
        
        # Test prediction
        user_id = sample_dataset.interactions['user_id'].iloc[0]
        predictions = model.predict(user_id)
        assert isinstance(predictions, dict)
        
        # Test recommendation
        recommendations = model.recommend(user_id, 5)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
    
    def test_popularity_baseline(self, sample_dataset):
        """Test popularity baseline model."""
        model = PopularityBaseline()
        
        # Test fitting
        model.fit(sample_dataset)
        assert model.is_fitted
        
        # Test prediction
        user_id = sample_dataset.interactions['user_id'].iloc[0]
        predictions = model.predict(user_id)
        assert isinstance(predictions, dict)
        
        # Test recommendation
        recommendations = model.recommend(user_id, 5)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5


class TestEvaluation:
    """Test cases for evaluation metrics."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        generator = SocialDataGenerator(seed=42)
        return generator.generate_dataset(n_users=20, n_items=10)
    
    def test_evaluator_init(self):
        """Test evaluator initialization."""
        evaluator = RecommendationEvaluator(k_values=[5, 10])
        assert evaluator.k_values == [5, 10]
    
    def test_data_split(self, sample_dataset):
        """Test data splitting functionality."""
        evaluator = RecommendationEvaluator()
        train_data, test_data = evaluator._split_data(sample_dataset, 0.2, 42)
        
        assert isinstance(train_data, SocialDataset)
        assert isinstance(test_data, SocialDataset)
        assert len(train_data.interactions) + len(test_data.interactions) == len(sample_dataset.interactions)
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        evaluator = RecommendationEvaluator()
        
        recommended = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant = ['item1', 'item3', 'item6']
        
        precision = evaluator._precision_at_k(recommended, relevant, 5)
        assert precision == 2/5  # 2 relevant items out of 5 recommended
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        evaluator = RecommendationEvaluator()
        
        recommended = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant = ['item1', 'item3', 'item6']
        
        recall = evaluator._recall_at_k(recommended, relevant, 5)
        assert recall == 2/3  # 2 relevant items found out of 3 total relevant
    
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        evaluator = RecommendationEvaluator()
        
        recommended = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant = ['item1', 'item3', 'item6']
        
        hit_rate = evaluator._hit_rate_at_k(recommended, relevant, 5)
        assert hit_rate == 1.0  # At least one relevant item found
    
    def test_model_evaluation(self, sample_dataset):
        """Test complete model evaluation."""
        model = PopularityBaseline()
        evaluator = RecommendationEvaluator(k_values=[5])
        
        metrics = evaluator.evaluate_model(model, sample_dataset, test_split=0.3)
        
        assert isinstance(metrics, dict)
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'precision@5' in metrics
        assert 'recall@5' in metrics


if __name__ == "__main__":
    pytest.main([__file__])
