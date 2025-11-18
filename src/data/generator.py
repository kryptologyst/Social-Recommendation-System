"""Data generation and loading utilities for social recommendation systems."""

import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass


@dataclass
class SocialDataset:
    """Container for social recommendation dataset."""
    
    interactions: pd.DataFrame
    social_network: Dict[str, List[str]]
    items: pd.DataFrame
    users: pd.DataFrame
    
    def __post_init__(self) -> None:
        """Validate dataset after initialization."""
        assert 'user_id' in self.interactions.columns
        assert 'item_id' in self.interactions.columns
        assert 'rating' in self.interactions.columns or 'interaction' in self.interactions.columns


class SocialDataGenerator:
    """Generate synthetic social recommendation datasets."""
    
    def __init__(self, seed: int = 42) -> None:
        """Initialize data generator with random seed.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_social_network(
        self, 
        n_users: int, 
        avg_friends: float = 3.0,
        clustering_coeff: float = 0.3
    ) -> Dict[str, List[str]]:
        """Generate a realistic social network using Watts-Strogatz model.
        
        Args:
            n_users: Number of users in the network.
            avg_friends: Average number of friends per user.
            clustering_coeff: Clustering coefficient (0-1).
            
        Returns:
            Dictionary mapping user_id to list of friend user_ids.
        """
        # Create Watts-Strogatz small world network
        k = int(avg_friends)  # Each node connected to k nearest neighbors
        p = 1 - clustering_coeff  # Rewiring probability
        
        G = nx.watts_strogatz_graph(n_users, k, p, seed=self.seed)
        
        # Convert to adjacency list
        social_network = {}
        for i in range(n_users):
            user_id = f"user_{i:04d}"
            friends = [f"user_{j:04d}" for j in G.neighbors(i)]
            social_network[user_id] = friends
            
        return social_network
    
    def generate_items(
        self, 
        n_items: int,
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate item metadata.
        
        Args:
            n_items: Number of items to generate.
            categories: List of item categories.
            
        Returns:
            DataFrame with item metadata.
        """
        if categories is None:
            categories = ['music', 'movie', 'book', 'game', 'food', 'travel', 'fashion', 'tech']
        
        items_data = []
        for i in range(n_items):
            item_id = f"item_{i:04d}"
            category = random.choice(categories)
            title = f"{category.title()} Item {i}"
            
            # Generate some text content for content-based features
            descriptions = {
                'music': f"Amazing {random.choice(['rock', 'pop', 'jazz', 'classical'])} album",
                'movie': f"Epic {random.choice(['action', 'drama', 'comedy', 'thriller'])} film",
                'book': f"Compelling {random.choice(['fiction', 'non-fiction', 'mystery', 'romance'])} novel",
                'game': f"Exciting {random.choice(['RPG', 'action', 'strategy', 'puzzle'])} game",
                'food': f"Delicious {random.choice(['Italian', 'Asian', 'Mexican', 'Mediterranean'])} cuisine",
                'travel': f"Beautiful {random.choice(['beach', 'mountain', 'city', 'countryside'])} destination",
                'fashion': f"Stylish {random.choice(['casual', 'formal', 'sporty', 'vintage'])} clothing",
                'tech': f"Advanced {random.choice(['smartphone', 'laptop', 'gadget', 'software'])} device"
            }
            
            description = descriptions.get(category, f"Great {category} item")
            
            items_data.append({
                'item_id': item_id,
                'title': title,
                'category': category,
                'description': description,
                'popularity_score': np.random.beta(2, 5),  # Skewed towards lower popularity
                'price': np.random.lognormal(3, 1),  # Log-normal price distribution
            })
        
        return pd.DataFrame(items_data)
    
    def generate_users(
        self, 
        n_users: int,
        demographics: bool = True
    ) -> pd.DataFrame:
        """Generate user metadata.
        
        Args:
            n_users: Number of users to generate.
            demographics: Whether to include demographic features.
            
        Returns:
            DataFrame with user metadata.
        """
        users_data = []
        for i in range(n_users):
            user_id = f"user_{i:04d}"
            
            user_data = {
                'user_id': user_id,
                'age': np.random.randint(18, 65),
                'activity_level': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2]),
            }
            
            if demographics:
                user_data.update({
                    'gender': np.random.choice(['M', 'F', 'Other'], p=[0.4, 0.4, 0.2]),
                    'location': np.random.choice(['US', 'EU', 'Asia', 'Other'], p=[0.4, 0.3, 0.2, 0.1]),
                    'income_level': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2]),
                })
            
            users_data.append(user_data)
        
        return pd.DataFrame(users_data)
    
    def generate_interactions(
        self,
        n_users: int,
        n_items: int,
        social_network: Dict[str, List[str]],
        items_df: pd.DataFrame,
        users_df: pd.DataFrame,
        interaction_rate: float = 0.1,
        social_influence: float = 0.3
    ) -> pd.DataFrame:
        """Generate user-item interactions with social influence.
        
        Args:
            n_users: Number of users.
            n_items: Number of items.
            social_network: Social network structure.
            items_df: Item metadata.
            users_df: User metadata.
            interaction_rate: Probability of user-item interaction.
            social_influence: Strength of social influence on interactions.
            
        Returns:
            DataFrame with user-item interactions.
        """
        interactions = []
        
        # Create user-item preference matrix
        user_preferences = np.random.rand(n_users, n_items)
        
        # Add social influence
        for user_idx, user_id in enumerate([f"user_{i:04d}" for i in range(n_users)]):
            friends = social_network.get(user_id, [])
            
            for friend_id in friends:
                friend_idx = int(friend_id.split('_')[1])
                # Social influence: user's preferences influenced by friends
                influence = np.random.rand(n_items) * social_influence
                user_preferences[user_idx] += influence * user_preferences[friend_idx]
        
        # Normalize preferences
        user_preferences = np.clip(user_preferences, 0, 1)
        
        # Generate interactions based on preferences
        for user_idx in range(n_users):
            user_id = f"user_{user_idx:04d}"
            
            for item_idx in range(n_items):
                item_id = f"item_{item_idx:04d}"
                
                # Interaction probability based on preference and item popularity
                item_popularity = items_df.iloc[item_idx]['popularity_score']
                preference = user_preferences[user_idx, item_idx]
                
                # Combine preference and popularity
                interaction_prob = preference * (1 + item_popularity) * interaction_rate
                
                if np.random.random() < interaction_prob:
                    # Generate rating (1-5 scale)
                    base_rating = int(preference * 4) + 1
                    rating = min(5, max(1, base_rating + np.random.randint(-1, 2)))
                    
                    # Add timestamp (simulate over 1 year)
                    timestamp = np.random.randint(0, 365 * 24 * 60 * 60)  # seconds in a year
                    
                    interactions.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'rating': rating,
                        'timestamp': timestamp,
                        'interaction_type': 'rating'
                    })
        
        return pd.DataFrame(interactions)
    
    def generate_dataset(
        self,
        n_users: int = 1000,
        n_items: int = 500,
        avg_friends: float = 5.0,
        interaction_rate: float = 0.15,
        social_influence: float = 0.3
    ) -> SocialDataset:
        """Generate complete social recommendation dataset.
        
        Args:
            n_users: Number of users.
            n_items: Number of items.
            avg_friends: Average number of friends per user.
            interaction_rate: Probability of user-item interaction.
            social_influence: Strength of social influence.
            
        Returns:
            Complete social recommendation dataset.
        """
        # Generate social network
        social_network = self.generate_social_network(n_users, avg_friends)
        
        # Generate items and users
        items_df = self.generate_items(n_items)
        users_df = self.generate_users(n_users)
        
        # Generate interactions
        interactions_df = self.generate_interactions(
            n_users, n_items, social_network, items_df, users_df,
            interaction_rate, social_influence
        )
        
        return SocialDataset(
            interactions=interactions_df,
            social_network=social_network,
            items=items_df,
            users=users_df
        )


def load_dataset_from_files(
    interactions_path: str,
    social_network_path: str,
    items_path: str,
    users_path: Optional[str] = None
) -> SocialDataset:
    """Load social recommendation dataset from CSV files.
    
    Args:
        interactions_path: Path to interactions CSV file.
        social_network_path: Path to social network CSV file.
        items_path: Path to items CSV file.
        users_path: Optional path to users CSV file.
        
    Returns:
        Loaded social recommendation dataset.
    """
    interactions_df = pd.read_csv(interactions_path)
    items_df = pd.read_csv(items_path)
    
    # Load social network
    social_df = pd.read_csv(social_network_path)
    social_network = {}
    for _, row in social_df.iterrows():
        user_id = row['user_id']
        friends = row['friends'].split(',') if pd.notna(row['friends']) else []
        social_network[user_id] = friends
    
    # Load users if provided
    if users_path:
        users_df = pd.read_csv(users_path)
    else:
        # Create minimal users dataframe
        unique_users = interactions_df['user_id'].unique()
        users_df = pd.DataFrame({'user_id': unique_users})
    
    return SocialDataset(
        interactions=interactions_df,
        social_network=social_network,
        items=items_df,
        users=users_df
    )


def save_dataset(dataset: SocialDataset, output_dir: str) -> None:
    """Save social recommendation dataset to CSV files.
    
    Args:
        dataset: Social recommendation dataset.
        output_dir: Output directory path.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save interactions
    dataset.interactions.to_csv(f"{output_dir}/interactions.csv", index=False)
    
    # Save items
    dataset.items.to_csv(f"{output_dir}/items.csv", index=False)
    
    # Save users
    dataset.users.to_csv(f"{output_dir}/users.csv", index=False)
    
    # Save social network
    social_data = []
    for user_id, friends in dataset.social_network.items():
        social_data.append({
            'user_id': user_id,
            'friends': ','.join(friends) if friends else ''
        })
    
    social_df = pd.DataFrame(social_data)
    social_df.to_csv(f"{output_dir}/social_network.csv", index=False)
