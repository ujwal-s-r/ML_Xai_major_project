"""
Simple L2CS-Net test to check correct initialization
"""

import torch
from l2cs import Pipeline

print("Testing L2CS-Net initialization...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    # Try different initialization methods
    print("\n1. Trying default Pipeline initialization...")
    gaze_pipeline = Pipeline(
        weights=None,  # Let it use default weights
        arch='ResNet50',
        device=torch.device('cpu')  # Use CPU for testing
    )
    print("✓ Pipeline created with default weights")
    
except Exception as e:
    print(f"✗ Failed with default: {e}")
    
    try:
        print("\n2. Trying without weights parameter...")
        gaze_pipeline = Pipeline(
            arch='ResNet50',
            device=torch.device('cpu')
        )
        print("✓ Pipeline created without weights param")
        
    except Exception as e2:
        print(f"✗ Failed without weights: {e2}")
        
        try:
            print("\n3. Trying minimal Pipeline()...")
            gaze_pipeline = Pipeline()
            print("✓ Pipeline created with minimal params")
            
        except Exception as e3:
            print(f"✗ Failed minimal: {e3}")
            print("\nLet's check Pipeline signature...")
            import inspect
            print(inspect.signature(Pipeline.__init__))
