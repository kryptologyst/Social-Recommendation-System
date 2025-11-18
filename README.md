# Social Recommendation System

A production-ready social recommendation system that leverages users' social networks to provide personalized recommendations. This system implements multiple state-of-the-art algorithms including social collaborative filtering, social matrix factorization, and graph neural networks.

## Features

### **Advanced Social Recommendation Models**
- **Social Collaborative Filtering**: Incorporates friends' preferences into collaborative filtering
- **Social Matrix Factorization**: Extends matrix factorization with social regularization
- **Social Graph Neural Networks**: Deep learning approach using graph structure
- **Popularity Baseline**: Simple baseline for comparison

### **Comprehensive Evaluation Framework**
- **Rating Prediction**: RMSE, MAE for accuracy assessment
- **Ranking Quality**: Precision@K, Recall@K, NDCG@K, Hit Rate@K
- **Coverage & Diversity**: Item coverage, recommendation diversity, novelty metrics
- **Model Comparison**: Automated benchmarking across all models

### **Interactive Demo**
- Real-time model training and evaluation
- Personalized recommendations with social context
- Social network visualization
- Performance comparison dashboards

### **Research-Grade Implementation**
- Clean, modular code with comprehensive type hints
- Reproducible experiments with proper seeding
- Production-ready project structure
- Comprehensive test suite

## Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/kryptologyst/Social-Recommendation-System.git
cd social-recommendation-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the interactive demo**
```bash
streamlit run demo.py
```

### Basic Usage

```python
from src.data.generator import SocialDataGenerator
from src.models.models import SocialCollaborativeFiltering
from src.utils.evaluation import RecommendationEvaluator

# Generate synthetic data
generator = SocialDataGenerator(seed=42)
dataset = generator.generate_dataset(n_users=1000, n_items=500)

# Train model
model = SocialCollaborativeFiltering(social_weight=0.3)
model.fit(dataset)

# Generate recommendations
recommendations = model.recommend("user_0001", n_recommendations=10)
print(recommendations)

# Evaluate model
evaluator = RecommendationEvaluator()
metrics = evaluator.evaluate_model(model, dataset)
print(metrics)
```

## Project Structure

```
social-recommendation-system/
├── src/                          # Source code
│   ├── data/                     # Data generation and loading
│   │   ├── __init__.py
│   │   └── generator.py
│   ├── models/                   # Recommendation models
│   │   ├── __init__.py
│   │   └── models.py
│   └── utils/                    # Evaluation and utilities
│       ├── __init__.py
│       └── evaluation.py
├── configs/                      # Configuration files
│   └── config.yaml
├── scripts/                      # Training and utility scripts
│   └── train.py
├── tests/                        # Test suite
│   └── test_social_recommender.py
├── notebooks/                    # Jupyter notebooks (optional)
├── data/                         # Data directory
├── outputs/                      # Model outputs and results
├── demo.py                       # Streamlit demo application
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Data Schema

### Interactions
- `user_id`: Unique user identifier
- `item_id`: Unique item identifier  
- `rating`: User rating (1-5 scale)
- `timestamp`: Interaction timestamp
- `interaction_type`: Type of interaction (e.g., 'rating')

### Items
- `item_id`: Unique item identifier
- `title`: Item title
- `category`: Item category
- `description`: Item description
- `popularity_score`: Item popularity score (0-1)
- `price`: Item price

### Users
- `user_id`: Unique user identifier
- `age`: User age
- `gender`: User gender
- `location`: User location
- `activity_level`: User activity level
- `income_level`: User income level

### Social Network
- `user_id`: User identifier
- `friends`: Comma-separated list of friend user IDs

## Models

### Social Collaborative Filtering
Combines traditional collaborative filtering with social influence by incorporating friends' preferences into the recommendation process.

**Key Features:**
- Social weight parameter to control influence strength
- Cosine similarity for user-user relationships
- Friends' rating aggregation for social prediction

### Social Matrix Factorization
Extends matrix factorization with social regularization to ensure users with similar social connections have similar latent factors.

**Key Features:**
- Configurable number of latent factors
- Social regularization term
- Bias terms for users and items

### Social Graph Neural Network
Uses graph neural networks to model complex social relationships and user-item interactions in a unified framework.

**Key Features:**
- Graph convolutional layers
- Social and interaction edge modeling
- Deep learning approach for complex patterns

## Evaluation Metrics

### Rating Prediction
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error

### Ranking Quality
- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation

### Coverage & Diversity
- **Coverage**: Fraction of items that can be recommended
- **Diversity**: Average dissimilarity of recommendations
- **Novelty**: Inverse of average item popularity

## Configuration

The system uses YAML configuration files for easy parameter tuning:

```yaml
# Data generation parameters
data:
  n_users: 1000
  n_items: 500
  avg_friends: 5.0
  interaction_rate: 0.15
  social_influence: 0.3

# Model parameters
models:
  social_cf:
    social_weight: 0.3
  social_mf:
    n_factors: 50
    reg_param: 0.01
    social_reg: 0.1
```

## Training and Evaluation

### Command Line Interface

```bash
# Generate synthetic data
python scripts/train.py --generate-data --data-dir data

# Train and evaluate models
python scripts/train.py --train-models --data-dir data --output-dir outputs

# Both operations
python scripts/train.py --generate-data --train-models
```

### Programmatic Interface

```python
from scripts.train import generate_and_save_data, train_and_evaluate_models
import yaml

# Load configuration
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Generate data
generate_and_save_data(config, 'data')

# Train models
train_and_evaluate_models(config, 'data', 'outputs')
```

## Interactive Demo

The Streamlit demo provides a comprehensive interface for:

1. **Data Generation**: Configure and generate synthetic datasets
2. **Model Training**: Train multiple models with different parameters
3. **Evaluation**: Compare model performance across multiple metrics
4. **Recommendations**: Get personalized recommendations with social context
5. **Visualization**: Explore social networks and recommendation patterns

### Running the Demo

```bash
streamlit run demo.py
```

The demo will be available at `http://localhost:8501`

## Testing

Run the comprehensive test suite:

```bash
pytest tests/
```

Or run specific test modules:

```bash
pytest tests/test_social_recommender.py::TestSocialModels
```

## Dependencies

### Core Libraries
- `numpy>=1.24.0`: Numerical computing
- `pandas>=2.0.0`: Data manipulation
- `scikit-learn>=1.3.0`: Machine learning utilities
- `scipy>=1.10.0`: Scientific computing

### Recommendation Libraries
- `implicit>=0.7.0`: Implicit feedback algorithms
- `lightfm>=1.17`: Hybrid recommendation algorithms
- `surprise>=1.1.3`: Collaborative filtering algorithms
- `networkx>=3.1`: Graph algorithms
- `torch>=2.0.0`: Deep learning framework
- `torch-geometric>=2.3.0`: Graph neural networks

### Visualization & Demo
- `streamlit>=1.25.0`: Interactive web applications
- `plotly>=5.15.0`: Interactive visualizations
- `matplotlib>=3.7.0`: Static visualizations

### Development
- `pytest>=7.4.0`: Testing framework
- `black>=23.0.0`: Code formatting
- `ruff>=0.0.280`: Fast Python linter
- `mypy>=1.5.0`: Static type checking

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
black src/ tests/
ruff check src/ tests/

# Run type checking
mypy src/

# Run tests
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{social_recommendation_system,
  title={Social Recommendation System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Social-Recommendation-System}
}
```

## Acknowledgments

- Built with modern Python best practices
- Implements state-of-the-art social recommendation algorithms
- Comprehensive evaluation framework
- Production-ready code structure
# Social-Recommendation-System
