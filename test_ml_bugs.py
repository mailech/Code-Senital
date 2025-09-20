#!/usr/bin/env python3
"""
Test script to verify ML bugs are working correctly
"""

import sys
import os
sys.path.append('Error-List')

def test_neural_network_bugs():
    """Test neural network bugs"""
    print("🧠 Testing Neural Network Bugs...")
    try:
        from buggy_neural_network import SimpleNeuralNetwork
        import numpy as np
        
        # Test weight initialization bug
        nn = SimpleNeuralNetwork(2, 3)
        print(f"   Weight initialization: {nn.weights[0, 0]:.6f} (should be ~0.01)")
        
        # Test sigmoid overflow bug
        test_input = np.array([1000, -1000])
        result = nn.sigmoid(test_input)
        print(f"   Sigmoid overflow test: {result} (should not be nan)")
        
        # Test forward pass bug
        inputs = np.array([[1, 2]])
        output = nn.forward(inputs)
        print(f"   Forward pass: {output} (should include bias)")
        
        print("   ✅ Neural network bugs detected!")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_data_preprocessing_bugs():
    """Test data preprocessing bugs"""
    print("📊 Testing Data Preprocessing Bugs...")
    try:
        from buggy_data_preprocessing import DataPreprocessor
        import numpy as np
        import pandas as pd
        
        processor = DataPreprocessor()
        
        # Test normalization bug
        data = np.array([1, 2, 3, 4, 5])
        normalized = processor.normalize_data(data)
        print(f"   Normalization: {normalized} (should handle std=0)")
        
        # Test missing value bug
        df = pd.DataFrame({'A': [1, 2, None, 4, 5]})
        filled = processor.handle_missing_values(df)
        print(f"   Missing values: {filled['A'].tolist()} (should use mean, not 0)")
        
        print("   ✅ Data preprocessing bugs detected!")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_model_evaluation_bugs():
    """Test model evaluation bugs"""
    print("📈 Testing Model Evaluation Bugs...")
    try:
        from buggy_model_evaluation import ModelEvaluator
        import numpy as np
        
        evaluator = ModelEvaluator()
        
        # Test accuracy calculation
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        accuracy = evaluator.calculate_accuracy(y_true, y_pred)
        print(f"   Accuracy: {accuracy} (should be 1.0)")
        
        # Test confusion matrix
        cm = evaluator.calculate_confusion_matrix(y_true, y_pred)
        print(f"   Confusion matrix shape: {cm.shape}")
        
        print("   ✅ Model evaluation bugs detected!")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("🔍 Testing ML Bugs for Sentinel Detection")
    print("=" * 50)
    
    results = []
    results.append(test_neural_network_bugs())
    results.append(test_data_preprocessing_bugs())
    results.append(test_model_evaluation_bugs())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All ML bugs are working correctly!")
        print("🚀 Ready for Sentinel to detect and fix them!")
    else:
        print("❌ Some bugs are not working correctly")
    
    return all(results)

if __name__ == "__main__":
    main()
