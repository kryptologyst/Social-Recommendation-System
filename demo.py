"""Streamlit demo for social recommendation system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from typing import Dict, List, Tuple, Optional

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.generator import SocialDataGenerator, SocialDataset
from models.models import (
    SocialCollaborativeFiltering, 
    SocialMatrixFactorization, 
    SocialGraphNeuralNetwork,
    PopularityBaseline
)
from utils.evaluation import RecommendationEvaluator, compare_models


def load_config() -> Dict:
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None


def generate_dataset(config: Dict) -> SocialDataset:
    """Generate synthetic social recommendation dataset."""
    generator = SocialDataGenerator(seed=config['data']['random_seed'])
    
    dataset = generator.generate_dataset(
        n_users=config['data']['n_users'],
        n_items=config['data']['n_items'],
        avg_friends=config['data']['avg_friends'],
        interaction_rate=config['data']['interaction_rate'],
        social_influence=config['data']['social_influence']
    )
    
    return dataset


def train_models(dataset: SocialDataset, config: Dict) -> Dict:
    """Train all recommendation models."""
    models = {}
    
    # Initialize models
    models['Popularity Baseline'] = PopularityBaseline()
    models['Social Collaborative Filtering'] = SocialCollaborativeFiltering(
        social_weight=config['models']['social_cf']['social_weight']
    )
    models['Social Matrix Factorization'] = SocialMatrixFactorization(
        n_factors=config['models']['social_mf']['n_factors'],
        reg_param=config['models']['social_mf']['reg_param'],
        social_reg=config['models']['social_mf']['social_reg'],
        n_iterations=config['models']['social_mf']['n_iterations']
    )
    models['Social Graph Neural Network'] = SocialGraphNeuralNetwork(
        hidden_dim=config['models']['social_gnn']['hidden_dim'],
        n_layers=config['models']['social_gnn']['n_layers'],
        dropout=config['models']['social_gnn']['dropout']
    )
    
    # Train models
    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            model.fit(dataset)
    
    return models


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Social Recommendation System",
        page_icon="👥",
        layout="wide"
    )
    
    st.title("👥 Social Recommendation System")
    st.markdown("A modern social recommendation system using friends' preferences and graph-based approaches")
    
    # Load configuration
    config = load_config()
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for controls
    st.sidebar.header("Configuration")
    
    # Data generation controls
    st.sidebar.subheader("Data Generation")
    n_users = st.sidebar.slider("Number of Users", 100, 2000, config['data']['n_users'])
    n_items = st.sidebar.slider("Number of Items", 100, 1000, config['data']['n_items'])
    avg_friends = st.sidebar.slider("Average Friends per User", 2.0, 10.0, config['data']['avg_friends'])
    social_influence = st.sidebar.slider("Social Influence", 0.0, 1.0, config['data']['social_influence'])
    
    # Generate dataset button
    if st.sidebar.button("Generate Dataset", type="primary"):
        with st.spinner("Generating dataset..."):
            st.session_state.dataset = generate_dataset({
                'data': {
                    'n_users': n_users,
                    'n_items': n_items,
                    'avg_friends': avg_friends,
                    'social_influence': social_influence,
                    'random_seed': config['data']['random_seed']
                }
            })
            st.session_state.models = {}
            st.session_state.evaluation_results = None
        st.success("Dataset generated successfully!")
    
    # Main content
    if st.session_state.dataset is not None:
        dataset = st.session_state.dataset
        
        # Dataset overview
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Users", len(dataset.users))
        with col2:
            st.metric("Items", len(dataset.items))
        with col3:
            st.metric("Interactions", len(dataset.interactions))
        with col4:
            avg_friends_actual = np.mean([len(friends) for friends in dataset.social_network.values()])
            st.metric("Avg Friends", f"{avg_friends_actual:.1f}")
        
        # Dataset statistics
        st.subheader("Dataset Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            rating_counts = dataset.interactions['rating'].value_counts().sort_index()
            fig_ratings = px.bar(
                x=rating_counts.index, 
                y=rating_counts.values,
                title="Rating Distribution",
                labels={'x': 'Rating', 'y': 'Count'}
            )
            st.plotly_chart(fig_ratings, use_container_width=True)
        
        with col2:
            # Category distribution
            category_counts = dataset.items['category'].value_counts()
            fig_categories = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Item Category Distribution"
            )
            st.plotly_chart(fig_categories, use_container_width=True)
        
        # Social network visualization
        st.subheader("Social Network")
        
        # Sample social network for visualization
        sample_users = list(dataset.social_network.keys())[:50]  # Show first 50 users
        sample_network = {k: v for k, v in dataset.social_network.items() if k in sample_users}
        
        # Create network data for visualization
        edges = []
        for user, friends in sample_network.items():
            for friend in friends:
                if friend in sample_users:
                    edges.append((user, friend))
        
        if edges:
            # Create a simple network visualization
            st.write(f"Showing social network for {len(sample_users)} users with {len(edges)} connections")
            
            # Network statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodes", len(sample_users))
            with col2:
                st.metric("Edges", len(edges))
            with col3:
                clustering_coeff = len(edges) / (len(sample_users) * (len(sample_users) - 1) / 2)
                st.metric("Density", f"{clustering_coeff:.3f}")
        
        # Model training and evaluation
        st.header("🤖 Model Training & Evaluation")
        
        if st.button("Train All Models", type="primary"):
            with st.spinner("Training models..."):
                st.session_state.models = train_models(dataset, config)
            st.success("All models trained successfully!")
        
        if st.session_state.models:
            # Model comparison
            st.subheader("Model Comparison")
            
            if st.button("Run Evaluation"):
                with st.spinner("Evaluating models..."):
                    evaluator = RecommendationEvaluator(k_values=config['evaluation']['k_values'])
                    results = compare_models(list(st.session_state.models.values()), dataset, evaluator)
                    st.session_state.evaluation_results = results
                st.success("Evaluation completed!")
            
            if st.session_state.evaluation_results is not None:
                results_df = st.session_state.evaluation_results
                
                # Display results table
                st.dataframe(results_df, use_container_width=True)
                
                # Visualization of results
                st.subheader("Performance Comparison")
                
                # Select metrics to visualize
                metric_cols = [col for col in results_df.columns if col != 'model']
                selected_metrics = st.multiselect(
                    "Select metrics to visualize",
                    metric_cols,
                    default=metric_cols[:4]
                )
                
                if selected_metrics:
                    # Create subplots
                    fig = make_subplots(
                        rows=1, 
                        cols=len(selected_metrics),
                        subplot_titles=selected_metrics
                    )
                    
                    for i, metric in enumerate(selected_metrics):
                        fig.add_trace(
                            go.Bar(
                                x=results_df['model'],
                                y=results_df[metric],
                                name=metric,
                                showlegend=False
                            ),
                            row=1, col=i+1
                        )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Interactive recommendations
        st.header("🎯 Interactive Recommendations")
        
        if st.session_state.models:
            # User selection
            user_id = st.selectbox(
                "Select a user",
                options=sorted(dataset.interactions['user_id'].unique()),
                index=0
            )
            
            # Model selection
            model_name = st.selectbox(
                "Select a model",
                options=list(st.session_state.models.keys()),
                index=1  # Default to Social Collaborative Filtering
            )
            
            model = st.session_state.models[model_name]
            
            # Generate recommendations
            n_recs = st.slider("Number of recommendations", 5, 20, config['demo']['n_recommendations'])
            
            if st.button("Generate Recommendations"):
                recommendations = model.recommend(user_id, n_recs)
                
                if recommendations:
                    st.subheader(f"Recommendations for {user_id} using {model_name}")
                    
                    # Display recommendations
                    rec_data = []
                    for i, (item_id, score) in enumerate(recommendations, 1):
                        item_info = dataset.items[dataset.items['item_id'] == item_id].iloc[0]
                        rec_data.append({
                            'Rank': i,
                            'Item ID': item_id,
                            'Title': item_info['title'],
                            'Category': item_info['category'],
                            'Score': f"{score:.3f}",
                            'Description': item_info['description']
                        })
                    
                    rec_df = pd.DataFrame(rec_data)
                    st.dataframe(rec_df, use_container_width=True)
                    
                    # Show user's friends and their preferences
                    st.subheader("Social Context")
                    
                    friends = dataset.social_network.get(user_id, [])
                    if friends:
                        st.write(f"**Friends of {user_id}:** {', '.join(friends)}")
                        
                        # Show friends' preferences for recommended items
                        friends_prefs = []
                        for friend_id in friends:
                            friend_interactions = dataset.interactions[
                                dataset.interactions['user_id'] == friend_id
                            ]
                            for _, rec in enumerate(recommendations[:5]):  # Show top 5
                                item_id = rec[0]
                                friend_rating = friend_interactions[
                                    friend_interactions['item_id'] == item_id
                                ]['rating']
                                
                                if not friend_rating.empty:
                                    friends_prefs.append({
                                        'Friend': friend_id,
                                        'Item': item_id,
                                        'Rating': friend_rating.iloc[0]
                                    })
                        
                        if friends_prefs:
                            st.write("**Friends' ratings for recommended items:**")
                            friends_df = pd.DataFrame(friends_prefs)
                            st.dataframe(friends_df, use_container_width=True)
                    else:
                        st.write(f"**{user_id} has no friends in the social network**")
                    
                    # Item similarity analysis
                    st.subheader("Item Similarity Analysis")
                    
                    if len(recommendations) >= 2:
                        # Show similarity between top recommendations
                        top_items = [item_id for item_id, _ in recommendations[:3]]
                        
                        similarity_data = []
                        for i, item1 in enumerate(top_items):
                            for j, item2 in enumerate(top_items[i+1:], i+1):
                                cat1 = dataset.items[dataset.items['item_id'] == item1]['category'].iloc[0]
                                cat2 = dataset.items[dataset.items['item_id'] == item2]['category'].iloc[0]
                                
                                similarity_data.append({
                                    'Item 1': item1,
                                    'Item 2': item2,
                                    'Category 1': cat1,
                                    'Category 2': cat2,
                                    'Same Category': 'Yes' if cat1 == cat2 else 'No'
                                })
                        
                        if similarity_data:
                            similarity_df = pd.DataFrame(similarity_data)
                            st.dataframe(similarity_df, use_container_width=True)
                else:
                    st.warning("No recommendations generated for this user.")
        else:
            st.info("Please train models first to generate recommendations.")
    
    else:
        st.info("👆 Please generate a dataset using the sidebar controls to get started.")
        
        # Show example of what the system can do
        st.header("🚀 What This System Does")
        
        st.markdown("""
        This social recommendation system demonstrates several key concepts:
        
        ### 🧠 **Social Recommendation Models**
        - **Social Collaborative Filtering**: Uses friends' preferences to improve recommendations
        - **Social Matrix Factorization**: Incorporates social relationships into matrix factorization
        - **Social Graph Neural Networks**: Leverages graph structure for deep social modeling
        - **Popularity Baseline**: Simple popularity-based recommendations for comparison
        
        ### 📊 **Comprehensive Evaluation**
        - **Rating Prediction**: RMSE, MAE for accuracy
        - **Ranking Quality**: Precision@K, Recall@K, NDCG@K, Hit Rate@K
        - **Coverage & Diversity**: Item coverage, recommendation diversity, novelty
        
        ### 🎯 **Interactive Features**
        - Generate synthetic social networks with realistic properties
        - Train and compare multiple models
        - Get personalized recommendations with social context
        - Visualize social influence on recommendations
        
        ### 🔬 **Research-Grade Implementation**
        - Clean, modular code with type hints
        - Comprehensive evaluation framework
        - Reproducible experiments with proper seeding
        - Production-ready structure
        """)


if __name__ == "__main__":
    main()
