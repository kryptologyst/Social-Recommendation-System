"""Main training script for social recommendation system."""

import argparse
import yaml
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.generator import SocialDataGenerator, save_dataset
from models.models import (
    SocialCollaborativeFiltering,
    SocialMatrixFactorization, 
    SocialGraphNeuralNetwork,
    PopularityBaseline
)
from utils.evaluation import RecommendationEvaluator, compare_models


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_and_save_data(config: Dict, output_dir: str) -> None:
    """Generate and save synthetic dataset."""
    print("Generating synthetic social recommendation dataset...")
    
    generator = SocialDataGenerator(seed=config['data']['random_seed'])
    
    dataset = generator.generate_dataset(
        n_users=config['data']['n_users'],
        n_items=config['data']['n_items'],
        avg_friends=config['data']['avg_friends'],
        interaction_rate=config['data']['interaction_rate'],
        social_influence=config['data']['social_influence']
    )
    
    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    save_dataset(dataset, output_dir)
    
    print(f"Dataset saved to {output_dir}")
    print(f"Users: {len(dataset.users)}")
    print(f"Items: {len(dataset.items)}")
    print(f"Interactions: {len(dataset.interactions)}")
    print(f"Average friends per user: {np.mean([len(friends) for friends in dataset.social_network.values()]):.2f}")


def train_and_evaluate_models(config: Dict, data_dir: str, output_dir: str) -> None:
    """Train and evaluate all models."""
    print("Training and evaluating models...")
    
    # Load dataset
    from data.generator import load_dataset_from_files
    dataset = load_dataset_from_files(
        interactions_path=f"{data_dir}/interactions.csv",
        social_network_path=f"{data_dir}/social_network.csv",
        items_path=f"{data_dir}/items.csv",
        users_path=f"{data_dir}/users.csv"
    )
    
    # Initialize models
    models = {
        'PopularityBaseline': PopularityBaseline(),
        'SocialCollaborativeFiltering': SocialCollaborativeFiltering(
            social_weight=config['models']['social_cf']['social_weight']
        ),
        'SocialMatrixFactorization': SocialMatrixFactorization(
            n_factors=config['models']['social_mf']['n_factors'],
            reg_param=config['models']['social_mf']['reg_param'],
            social_reg=config['models']['social_mf']['social_reg'],
            n_iterations=config['models']['social_mf']['n_iterations']
        ),
        'SocialGraphNeuralNetwork': SocialGraphNeuralNetwork(
            hidden_dim=config['models']['social_gnn']['hidden_dim'],
            n_layers=config['models']['social_gnn']['n_layers'],
            dropout=config['models']['social_gnn']['dropout']
        )
    }
    
    # Evaluate models
    evaluator = RecommendationEvaluator(k_values=config['evaluation']['k_values'])
    results = compare_models(list(models.values()), dataset, evaluator)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    print("Model evaluation completed!")
    print("\nResults:")
    print(results.to_string(index=False))
    
    # Save detailed results
    detailed_results = []
    for model_name, model in models.items():
        print(f"\nGenerating recommendations for {model_name}...")
        
        # Sample users for recommendations
        sample_users = dataset.interactions['user_id'].unique()[:10]
        
        for user_id in sample_users:
            recommendations = model.recommend(user_id, 10)
            for rank, (item_id, score) in enumerate(recommendations, 1):
                detailed_results.append({
                    'model': model_name,
                    'user_id': user_id,
                    'item_id': item_id,
                    'rank': rank,
                    'score': score
                })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(f"{output_dir}/detailed_recommendations.csv", index=False)
    
    print(f"Results saved to {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train social recommendation models")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory to save/load data")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Directory to save outputs")
    parser.add_argument("--generate-data", action="store_true",
                       help="Generate new synthetic data")
    parser.add_argument("--train-models", action="store_true",
                       help="Train and evaluate models")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seeds
    np.random.seed(config['data']['random_seed'])
    
    if args.generate_data:
        generate_and_save_data(config, args.data_dir)
    
    if args.train_models:
        train_and_evaluate_models(config, args.data_dir, args.output_dir)
    
    if not args.generate_data and not args.train_models:
        print("Please specify --generate-data and/or --train-models")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
