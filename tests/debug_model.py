"""
Debug script to inspect the ViT model architecture and find attention layers.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from context.emotion_analyzer import EmotionAnalyzer
from PIL import Image
import numpy as np
import cv2

print("\n" + "="*60)
print("MODEL ARCHITECTURE INSPECTION")
print("="*60)

# Initialize analyzer
analyzer = EmotionAnalyzer(enable_face_detector=True)
model = analyzer.model

print("\n1. MODEL TYPE AND STRUCTURE:")
print(f"Model class: {type(model).__name__}")
print(f"Model config type: {type(model.config).__name__}")

print("\n2. MODEL ATTRIBUTES:")
for attr in dir(model):
    if not attr.startswith('_'):
        print(f"  - {attr}")

print("\n3. CHECKING FOR VIT/VISION MODEL:")
has_vit = hasattr(model, 'vit')
has_vision = hasattr(model, 'vision_model')
has_base = hasattr(model, 'base_model')
has_encoder = hasattr(model, 'encoder')

print(f"  has 'vit': {has_vit}")
print(f"  has 'vision_model': {has_vision}")
print(f"  has 'base_model': {has_base}")
print(f"  has 'encoder': {has_encoder}")

print("\n4. MODEL NAMED MODULES:")
print("First 30 module names:")
for i, (name, module) in enumerate(model.named_modules()):
    if i < 30:
        print(f"  {i}: {name} -> {type(module).__name__}")

print("\n5. CHECKING MODEL CONFIG:")
config = model.config
print(f"Config attributes:")
for attr in dir(config):
    if not attr.startswith('_') and not callable(getattr(config, attr)):
        try:
            val = getattr(config, attr)
            if not isinstance(val, (dict, list)) or len(str(val)) < 100:
                print(f"  {attr}: {val}")
        except:
            pass

print("\n6. TESTING FORWARD PASS WITH output_attentions:")
try:
    # Create dummy input
    dummy_image = Image.new('RGB', (224, 224), color='red')
    inputs = analyzer.processor(images=dummy_image, return_tensors="pt")
    inputs = {k: v.to(analyzer.device) for k, v in inputs.items()}
    
    print(f"Input keys: {inputs.keys()}")
    print(f"Input shape: {inputs['pixel_values'].shape}")
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    print(f"\nOutput type: {type(outputs)}")
    print(f"Output attributes:")
    for attr in dir(outputs):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    print(f"\nChecking output.attentions:")
    if hasattr(outputs, 'attentions'):
        attentions = outputs.attentions
        print(f"  attentions type: {type(attentions)}")
        print(f"  attentions is None: {attentions is None}")
        if attentions is not None:
            print(f"  attentions length: {len(attentions)}")
            if len(attentions) > 0:
                print(f"  First attention shape: {attentions[0].shape}")
                print(f"  Last attention shape: {attentions[-1].shape}")
        else:
            print("  ❌ ATTENTIONS IS NONE - THIS IS THE PROBLEM")
    else:
        print("  ❌ NO 'attentions' ATTRIBUTE")
    
    print(f"\nChecking output.hidden_states:")
    if hasattr(outputs, 'hidden_states'):
        hidden_states = outputs.hidden_states
        print(f"  hidden_states type: {type(hidden_states)}")
        print(f"  hidden_states is None: {hidden_states is None}")
    
except Exception as e:
    print(f"\n❌ ERROR during forward pass: {e}")
    import traceback
    traceback.print_exc()

print("\n7. SEARCHING FOR ATTENTION MODULES:")
attention_modules = []
for name, module in model.named_modules():
    if 'attention' in name.lower() or 'attn' in name.lower():
        attention_modules.append((name, type(module).__name__))

print(f"Found {len(attention_modules)} attention-related modules:")
for name, mtype in attention_modules[:20]:
    print(f"  {name} -> {mtype}")

print("\n8. CHECKING MODEL FORWARD SIGNATURE:")
import inspect
sig = inspect.signature(model.forward)
print(f"Forward parameters: {list(sig.parameters.keys())}")

print("\n" + "="*60)
print("INSPECTION COMPLETE")
print("="*60 + "\n")
