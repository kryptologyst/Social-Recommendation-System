"""Evaluation metrics and utilities for social recommendation systems."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict

from src.data.generator import SocialDataset
from src.models import BaseSocialRecommender


class RecommendationEvaluator:
    """Evaluator for social recommendation models."""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]) -> None:
        """Initialize evaluator.
        
        Args:
            k_values: List of k values for top-k metrics.
        """
        self.k_values = k_values
    
    def evaluate_model(
        self, 
        model: BaseSocialRecommender, 
        dataset: SocialDataset,
        test_split: float = 0.2,
        random_seed: int = 42
    ) -> Dict[str, float]:
        """Evaluate a recommendation model.
        
        Args:
            model: Trained recommendation model.
            dataset: Social recommendation dataset.
            test_split: Fraction of data to use for testing.
            random_seed: Random seed for reproducibility.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Split data into train/test
        train_data, test_data = self._split_data(dataset, test_split, random_seed)
        
        # Retrain model on training data
        model.fit(train_data)
        
        # Evaluate on test data
        metrics = {}
        
        # Rating prediction metrics
        rating_metrics = self._evaluate_rating_prediction(model, test_data)
        metrics.update(rating_metrics)
        
        # Ranking metrics
        ranking_metrics = self._evaluate_ranking(model, test_data)
        metrics.update(ranking_metrics)
        
        # Coverage and diversity metrics
        coverage_metrics = self._evaluate_coverage_diversity(model, dataset)
        metrics.update(coverage_metrics)
        
        return metrics
    
    def _split_data(
        self, 
        dataset: SocialDataset, 
        test_split: float, 
        random_seed: int
    ) -> Tuple[SocialDataset, SocialDataset]:
        """Split dataset into train and test sets.
        
        Args:
            dataset: Original dataset.
            test_split: Fraction of data to use for testing.
            random_seed: Random seed for reproducibility.
            
        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        np.random.seed(random_seed)
        
        # Sort interactions by timestamp
        interactions = dataset.interactions.sort_values('timestamp')
        
        # Split interactions
        n_interactions = len(interactions)
        test_size = int(n_interactions * test_split)
        
        test_interactions = interactions.tail(test_size)
        train_interactions = interactions.head(n_interactions - test_size)
        
        # Create train dataset
        train_dataset = SocialDataset(
            interactions=train_interactions,
            social_network=dataset.social_network,
            items=dataset.items,
            users=dataset.users
        )
        
        # Create test dataset
        test_dataset = SocialDataset(
            interactions=test_interactions,
            social_network=dataset.social_network,
            items=dataset.items,
            users=dataset.users
        )
        
        return train_dataset, test_dataset
    
    def _evaluate_rating_prediction(
        self, 
        model: BaseSocialRecommender, 
        test_data: SocialDataset
    ) -> Dict[str, float]:
        """Evaluate rating prediction accuracy.
        
        Args:
            model: Trained recommendation model.
            test_data: Test dataset.
            
        Returns:
            Dictionary of rating prediction metrics.
        """
        predictions = []
        actuals = []
        
        for _, row in test_data.interactions.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            pred_dict = model.predict(user_id, [item_id])
            if item_id in pred_dict:
                predictions.append(pred_dict[item_id])
                actuals.append(actual_rating)
        
        if not predictions:
            return {'rmse': 0.0, 'mae': 0.0}
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        return {'rmse': rmse, 'mae': mae}
    
    def _evaluate_ranking(
        self, 
        model: BaseSocialRecommender, 
        test_data: SocialDataset
    ) -> Dict[str, float]:
        """Evaluate ranking quality.
        
        Args:
            model: Trained recommendation model.
            test_data: Test dataset.
            
        Returns:
            Dictionary of ranking metrics.
        """
        metrics = {}
        
        # Group test interactions by user
        user_interactions = defaultdict(list)
        for _, row in test_data.interactions.iterrows():
            user_interactions[row['user_id']].append((row['item_id'], row['rating']))
        
        # Compute metrics for each k
        for k in self.k_values:
            precisions = []
            recalls = []
            ndcgs = []
            hit_rates = []
            
            for user_id, interactions in user_interactions.items():
                if len(interactions) == 0:
                    continue
                
                # Get recommendations
                recommendations = model.recommend(user_id, k)
                if not recommendations:
                    continue
                
                recommended_items = [item_id for item_id, _ in recommendations]
                
                # Get relevant items (rating >= 4)
                relevant_items = [item_id for item_id, rating in interactions if rating >= 4]
                
                if not relevant_items:
                    continue
                
                # Compute metrics
                precision = self._precision_at_k(recommended_items, relevant_items, k)
                recall = self._recall_at_k(recommended_items, relevant_items, k)
                ndcg = self._ndcg_at_k(recommended_items, interactions, k)
                hit_rate = self._hit_rate_at_k(recommended_items, relevant_items, k)
                
                precisions.append(precision)
                recalls.append(recall)
                ndcgs.append(ndcg)
                hit_rates.append(hit_rate)
            
            # Average metrics
            metrics[f'precision@{k}'] = np.mean(precisions) if precisions else 0.0
            metrics[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0
            metrics[f'ndcg@{k}'] = np.mean(ndcgs) if ndcgs else 0.0
            metrics[f'hit_rate@{k}'] = np.mean(hit_rates) if hit_rates else 0.0
        
        return metrics
    
    def _precision_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """Compute precision at k."""
        if k == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        recommended_set = set(recommended_k)
        
        intersection = len(recommended_set.intersection(relevant_set))
        return intersection / k
    
    def _recall_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """Compute recall at k."""
        if not relevant:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        recommended_set = set(recommended_k)
        
        intersection = len(recommended_set.intersection(relevant_set))
        return intersection / len(relevant_set)
    
    def _ndcg_at_k(self, recommended: List[str], interactions: List[Tuple[str, float]], k: int) -> float:
        """Compute NDCG at k."""
        if k == 0:
            return 0.0
        
        # Create relevance scores
        relevance_scores = {item_id: rating for item_id, rating in interactions}
        
        # Compute DCG
        dcg = 0.0
        for i, item_id in enumerate(recommended[:k]):
            if item_id in relevance_scores:
                dcg += relevance_scores[item_id] / np.log2(i + 2)
        
        # Compute IDCG
        sorted_interactions = sorted(interactions, key=lambda x: x[1], reverse=True)
        idcg = 0.0
        for i, (item_id, rating) in enumerate(sorted_interactions[:k]):
            idcg += rating / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _hit_rate_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """Compute hit rate at k."""
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        recommended_set = set(recommended_k)
        
        intersection = len(recommended_set.intersection(relevant_set))
        return 1.0 if intersection > 0 else 0.0
    
    def _evaluate_coverage_diversity(
        self, 
        model: BaseSocialRecommender, 
        dataset: SocialDataset
    ) -> Dict[str, float]:
        """Evaluate coverage and diversity metrics.
        
        Args:
            model: Trained recommendation model.
            dataset: Social recommendation dataset.
            
        Returns:
            Dictionary of coverage and diversity metrics.
        """
        # Get all users
        users = dataset.interactions['user_id'].unique()
        
        # Generate recommendations for all users
        all_recommended_items = set()
        user_recommendations = {}
        
        for user_id in users:
            recommendations = model.recommend(user_id, 10)
            recommended_items = [item_id for item_id, _ in recommendations]
            user_recommendations[user_id] = recommended_items
            all_recommended_items.update(recommended_items)
        
        # Coverage: fraction of items that can be recommended
        total_items = len(dataset.items)
        coverage = len(all_recommended_items) / total_items if total_items > 0 else 0.0
        
        # Diversity: average pairwise dissimilarity of recommendations
        diversity = self._compute_diversity(user_recommendations, dataset)
        
        # Novelty: average popularity of recommended items
        novelty = self._compute_novelty(all_recommended_items, dataset)
        
        return {
            'coverage': coverage,
            'diversity': diversity,
            'novelty': novelty
        }
    
    def _compute_diversity(
        self, 
        user_recommendations: Dict[str, List[str]], 
        dataset: SocialDataset
    ) -> float:
        """Compute diversity of recommendations."""
        diversities = []
        
        for user_id, recommendations in user_recommendations.items():
            if len(recommendations) < 2:
                continue
            
            # Compute pairwise Jaccard dissimilarity
            total_dissimilarity = 0.0
            pairs = 0
            
            for i in range(len(recommendations)):
                for j in range(i + 1, len(recommendations)):
                    item1 = recommendations[i]
                    item2 = recommendations[j]
                    
                    # Get item categories
                    item1_cat = dataset.items[dataset.items['item_id'] == item1]['category'].iloc[0]
                    item2_cat = dataset.items[dataset.items['item_id'] == item2]['category'].iloc[0]
                    
                    # Dissimilarity is 1 if different categories, 0 if same
                    dissimilarity = 1.0 if item1_cat != item2_cat else 0.0
                    total_dissimilarity += dissimilarity
                    pairs += 1
            
            if pairs > 0:
                diversities.append(total_dissimilarity / pairs)
        
        return np.mean(diversities) if diversities else 0.0
    
    def _compute_novelty(
        self, 
        recommended_items: set, 
        dataset: SocialDataset
    ) -> float:
        """Compute novelty of recommendations."""
        if not recommended_items:
            return 0.0
        
        # Get popularity scores of recommended items
        recommended_items_df = dataset.items[dataset.items['item_id'].isin(recommended_items)]
        
        if len(recommended_items_df) == 0:
            return 0.0
        
        # Novelty is inverse of average popularity
        avg_popularity = recommended_items_df['popularity_score'].mean()
        novelty = 1.0 - avg_popularity
        
        return max(0.0, novelty)


def compare_models(
    models: List[BaseSocialRecommender],
    dataset: SocialDataset,
    evaluator: Optional[RecommendationEvaluator] = None
) -> pd.DataFrame:
    """Compare multiple recommendation models.
    
    Args:
        models: List of recommendation models to compare.
        dataset: Social recommendation dataset.
        evaluator: Optional evaluator instance.
        
    Returns:
        DataFrame with comparison results.
    """
    if evaluator is None:
        evaluator = RecommendationEvaluator()
    
    results = []
    
    for model in models:
        print(f"Evaluating {model.name}...")
        metrics = evaluator.evaluate_model(model, dataset)
        metrics['model'] = model.name
        results.append(metrics)
    
    return pd.DataFrame(results)
