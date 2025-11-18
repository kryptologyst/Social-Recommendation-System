#!/usr/bin/env python3
"""Quick test script for the social recommendation system."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_functionality():
    """Test basic functionality of the social recommendation system."""
    print("Testing Social Recommendation System")
    print("=" * 40)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from src.data.generator import SocialDataGenerator
        from src.models import SocialCollaborativeFiltering, PopularityBaseline
        from src.utils import RecommendationEvaluator
        print("   ✓ All imports successful")
        
        # Test data generation
        print("2. Testing data generation...")
        generator = SocialDataGenerator(seed=42)
        dataset = generator.generate_dataset(n_users=50, n_items=25)
        print(f"   ✓ Generated dataset: {len(dataset.users)} users, {len(dataset.items)} items")
        
        # Test model training
        print("3. Testing model training...")
        model = SocialCollaborativeFiltering(social_weight=0.3)
        model.fit(dataset)
        print("   ✓ Model trained successfully")
        
        # Test recommendations
        print("4. Testing recommendations...")
        user_id = dataset.interactions['user_id'].iloc[0]
        recommendations = model.recommend(user_id, 5)
        print(f"   ✓ Generated {len(recommendations)} recommendations for {user_id}")
        
        # Test evaluation
        print("5. Testing evaluation...")
        evaluator = RecommendationEvaluator(k_values=[5])
        metrics = evaluator.evaluate_model(model, dataset, test_split=0.3)
        print(f"   ✓ Evaluation completed: RMSE={metrics.get('rmse', 0):.3f}")
        
        print("\n" + "=" * 40)
        print("✓ All tests passed! System is working correctly.")
        print("\nNext steps:")
        print("- Run 'streamlit run demo.py' for the interactive demo")
        print("- Run 'python scripts/train.py --generate-data --train-models' for full training")
        print("- Check the README.md for detailed usage instructions")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
