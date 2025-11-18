"""Example notebook for social recommendation system."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from data.generator import SocialDataGenerator
from models.models import (
    SocialCollaborativeFiltering,
    SocialMatrixFactorization,
    PopularityBaseline
)
from utils.evaluation import RecommendationEvaluator

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Social Recommendation System - Example Usage")
print("=" * 50)

# 1. Generate synthetic data
print("\n1. Generating synthetic social recommendation dataset...")
generator = SocialDataGenerator(seed=42)
dataset = generator.generate_dataset(
    n_users=500,
    n_items=200,
    avg_friends=4.0,
    interaction_rate=0.2,
    social_influence=0.4
)

print(f"Dataset generated:")
print(f"  - Users: {len(dataset.users)}")
print(f"  - Items: {len(dataset.items)}")
print(f"  - Interactions: {len(dataset.interactions)}")
print(f"  - Avg friends per user: {np.mean([len(friends) for friends in dataset.social_network.values()]):.2f}")

# 2. Explore the data
print("\n2. Exploring the dataset...")

# Rating distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Rating distribution
rating_counts = dataset.interactions['rating'].value_counts().sort_index()
axes[0].bar(rating_counts.index, rating_counts.values)
axes[0].set_title('Rating Distribution')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Count')

# Category distribution
category_counts = dataset.items['category'].value_counts()
axes[1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
axes[1].set_title('Item Category Distribution')

plt.tight_layout()
plt.show()

# 3. Train models
print("\n3. Training recommendation models...")

models = {
    'Popularity Baseline': PopularityBaseline(),
    'Social Collaborative Filtering': SocialCollaborativeFiltering(social_weight=0.3),
    'Social Matrix Factorization': SocialMatrixFactorization(n_factors=20, n_iterations=50)
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(dataset)

print("All models trained successfully!")

# 4. Generate recommendations
print("\n4. Generating recommendations...")

# Select a sample user
sample_user = dataset.interactions['user_id'].iloc[0]
print(f"Generating recommendations for user: {sample_user}")

# Show user's friends
friends = dataset.social_network.get(sample_user, [])
print(f"User's friends: {friends}")

# Generate recommendations from each model
for name, model in models.items():
    recommendations = model.recommend(sample_user, 5)
    print(f"\n{name} recommendations:")
    for i, (item_id, score) in enumerate(recommendations, 1):
        item_info = dataset.items[dataset.items['item_id'] == item_id].iloc[0]
        print(f"  {i}. {item_info['title']} ({item_info['category']}) - Score: {score:.3f}")

# 5. Evaluate models
print("\n5. Evaluating models...")

evaluator = RecommendationEvaluator(k_values=[5, 10])
results = []

for name, model in models.items():
    print(f"Evaluating {name}...")
    metrics = evaluator.evaluate_model(model, dataset, test_split=0.2)
    metrics['model'] = name
    results.append(metrics)

# Create results dataframe
results_df = pd.DataFrame(results)
print("\nEvaluation Results:")
print(results_df.round(4))

# 6. Visualize results
print("\n6. Visualizing results...")

# Select key metrics for visualization
key_metrics = ['precision@5', 'recall@5', 'ndcg@5', 'coverage']
available_metrics = [m for m in key_metrics if m in results_df.columns]

if available_metrics:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics):
        if i < len(axes):
            bars = axes[i].bar(results_df['model'], results_df[metric])
            axes[i].set_title(f'{metric.replace("@", " @ ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
    
    # Hide unused subplots
    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# 7. Social influence analysis
print("\n7. Analyzing social influence...")

# Compare recommendations with and without social influence
cf_model = SocialCollaborativeFiltering(social_weight=0.0)  # No social influence
cf_social_model = SocialCollaborativeFiltering(social_weight=0.5)  # High social influence

cf_model.fit(dataset)
cf_social_model.fit(dataset)

# Get recommendations for a user with friends
user_with_friends = None
for user_id, friends in dataset.social_network.items():
    if len(friends) > 0:
        user_with_friends = user_id
        break

if user_with_friends:
    print(f"Comparing recommendations for user: {user_with_friends}")
    print(f"User's friends: {dataset.social_network[user_with_friends]}")
    
    recs_no_social = cf_model.recommend(user_with_friends, 5)
    recs_with_social = cf_social_model.recommend(user_with_friends, 5)
    
    print("\nRecommendations without social influence:")
    for i, (item_id, score) in enumerate(recs_no_social, 1):
        item_info = dataset.items[dataset.items['item_id'] == item_id].iloc[0]
        print(f"  {i}. {item_info['title']} - Score: {score:.3f}")
    
    print("\nRecommendations with social influence:")
    for i, (item_id, score) in enumerate(recs_with_social, 1):
        item_info = dataset.items[dataset.items['item_id'] == item_id].iloc[0]
        print(f"  {i}. {item_info['title']} - Score: {score:.3f}")

print("\n" + "=" * 50)
print("Example completed successfully!")
print("Run 'streamlit run demo.py' for the interactive demo.")
