"""
Emotion Processor
Wraps EmotionAnalyzer to provide continuous timeline data and summary metrics.
Includes XAI features: Attention Maps and Grad-CAM for interpretability.
"""

import sys
import base64
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch
import cv2
from collections import Counter

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from context.emotion_analyzer import EmotionAnalyzer


class EmotionProcessor:
    """
    High-level processor for emotion analysis.
    Generates continuous timeline data, aggregate summary, and XAI visualizations.
    """
    
    def __init__(self):
        """Initialize the emotion processor."""
        self.emotion_analyzer = EmotionAnalyzer(enable_face_detector=True)
        self.xai_frame_count = 4  # Extract XAI data for exactly 4 strategic frames
    
    def extract_attention_maps(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Extract attention maps from ViT model for XAI.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Dictionary with attention map data or None if failed
        """
        try:
            from PIL import Image
            
            # Detect and crop face
            face_crop, face_region = self.emotion_analyzer._detect_and_crop_face(frame)
            if face_crop is None:
                return None
            
            # Convert to PIL and preprocess
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_face)
            inputs = self.emotion_analyzer.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.emotion_analyzer.device) for k, v in inputs.items()}
            
            # Get model
            model = self.emotion_analyzer.model
            
            # SOLUTION: Replace SDPA attention with regular attention to capture weights
            # Save original attention implementation
            original_attentions = []
            
            # Access encoder layers and replace attention mechanism temporarily
            for layer_idx, layer in enumerate(model.vit.encoder.layer):
                original_attentions.append(layer.attention)
                
                # Replace ViTSdpaAttention with regular ViTAttention that returns weights
                # We'll use a hook-based approach instead
            
            # Use forward hook to capture attention weights
            attention_maps = []
            
            def capture_attention_hook(module, input, output):
                # For ViT attention layers, we need to manually compute attention
                # from query, key, value
                if hasattr(module, 'query') and hasattr(module, 'key'):
                    # This is ViTSdpaSelfAttention
                    hidden_states = input[0]
                    batch_size, seq_length, _ = hidden_states.size()
                    
                    # Compute Q, K, V
                    query = module.query(hidden_states)
                    key = module.key(hidden_states)
                    value = module.value(hidden_states)
                    
                    # Reshape for multi-head attention
                    num_heads = 12  # From config
                    head_dim = 768 // 12  # hidden_size // num_heads
                    
                    query = query.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
                    key = key.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
                    
                    # Compute attention weights: (batch, heads, seq_len, seq_len)
                    attention_scores = torch.matmul(query, key.transpose(-1, -2))
                    attention_scores = attention_scores / np.sqrt(head_dim)
                    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
                    
                    attention_maps.append(attention_probs.detach())
            
            # Register hooks on all attention layers
            hooks = []
            for layer in model.vit.encoder.layer:
                hook = layer.attention.attention.register_forward_hook(capture_attention_hook)
                hooks.append(hook)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Check if we captured attention
            if len(attention_maps) == 0:
                print("[EmotionProcessor] No attention maps captured")
                return self._create_gradient_attention_map(frame, inputs, face_crop, face_region)
            
            # Use last layer attention
            last_layer_attention = attention_maps[-1]  # Shape: (1, num_heads, seq_len, seq_len)
            
            # Average across all attention heads
            avg_attention = last_layer_attention.mean(dim=1)  # Shape: (1, seq_len, seq_len)
            
            # Get attention weights for CLS token (first token) attending to all patches
            cls_attention = avg_attention[0, 0, 1:]  # Exclude CLS token itself
            
            # Reshape to spatial grid
            num_patches = cls_attention.shape[0]
            grid_size = int(np.sqrt(num_patches))
            
            attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
            
            # Normalize to 0-1
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
            
            # Resize to original face crop size for overlay
            attention_resized = cv2.resize(attention_map, (face_crop.shape[1], face_crop.shape[0]))
            
            return {
                'attention_map': attention_resized,
                'grid_size': grid_size,
                'face_region': face_region,
                'face_crop': face_crop
            }
        
        except Exception as e:
            print(f"[EmotionProcessor] Attention extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_gradient_attention_map(self, frame: np.ndarray, inputs: Dict, face_crop: np.ndarray, face_region: Dict) -> Optional[Dict]:
        """
        Fallback: Create attention-like map using input gradients.
        
        Args:
            frame: Original frame
            inputs: Preprocessed inputs
            face_crop: Cropped face image
            face_region: Face region coordinates
            
        Returns:
            Dictionary with gradient-based attention map
        """
        try:
            # Enable gradients
            inputs['pixel_values'].requires_grad = True
            
            # Forward pass
            outputs = self.emotion_analyzer.model(**inputs)
            logits = outputs.logits
            
            # Get predicted class
            pred_class = logits.argmax(dim=1).item()
            
            # Backward pass
            self.emotion_analyzer.model.zero_grad()
            class_score = logits[0, pred_class]
            class_score.backward()
            
            # Get gradients
            gradients = inputs['pixel_values'].grad.data
            
            # Compute gradient magnitude as attention proxy
            grad_magnitude = torch.abs(gradients).mean(dim=1).squeeze().cpu().numpy()
            
            # Normalize
            attention_map = (grad_magnitude - grad_magnitude.min()) / (grad_magnitude.max() - grad_magnitude.min() + 1e-8)
            
            # Resize to face crop size
            attention_resized = cv2.resize(attention_map, (face_crop.shape[1], face_crop.shape[0]))
            
            return {
                'attention_map': attention_resized,
                'grid_size': attention_map.shape[0],
                'face_region': face_region,
                'face_crop': face_crop,
                'method': 'gradient_fallback'
            }
        
        except Exception as e:
            print(f"[EmotionProcessor] Gradient fallback error: {e}")
            return None
    
    def extract_gradcam(self, frame: np.ndarray, target_emotion: Optional[str] = None) -> Optional[Dict]:
        """
        Extract Grad-CAM heatmap for XAI.
        
        Args:
            frame: Input frame (BGR)
            target_emotion: Target emotion class for Grad-CAM (None = use predicted class)
            
        Returns:
            Dictionary with Grad-CAM data or None if failed
        """
        try:
            from PIL import Image
            
            # Detect and crop face
            face_crop, face_region = self.emotion_analyzer._detect_and_crop_face(frame)
            if face_crop is None:
                return None
            
            # Convert to PIL and preprocess
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_face)
            inputs = self.emotion_analyzer.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.emotion_analyzer.device) for k, v in inputs.items()}
            
            # Enable gradients
            inputs['pixel_values'].requires_grad = True
            
            # Forward pass
            outputs = self.emotion_analyzer.model(**inputs)
            logits = outputs.logits
            
            # Get target class
            if target_emotion is None:
                target_class = logits.argmax(dim=1).item()
            else:
                # Map emotion name to class index
                label2id = {v.lower(): k for k, v in self.emotion_analyzer.model.config.id2label.items()}
                target_class = label2id.get(target_emotion.lower(), 0)
            
            # Backward pass for target class
            self.emotion_analyzer.model.zero_grad()
            class_score = logits[0, target_class]
            class_score.backward()
            
            # Get gradients
            gradients = inputs['pixel_values'].grad.data
            
            # Global average pooling of gradients
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            
            # Get activations from last conv layer
            # For ViT, we use the patch embeddings
            activations = inputs['pixel_values'].detach()
            
            # Weight activations by gradients
            for i in range(activations.shape[1]):
                activations[:, i, :, :] *= pooled_gradients[i]
            
            # Average across channels
            heatmap = torch.mean(activations, dim=1).squeeze()
            
            # ReLU and normalize
            heatmap = torch.clamp(heatmap, min=0)
            heatmap = heatmap / (torch.max(heatmap) + 1e-8)
            heatmap = heatmap.cpu().numpy()
            
            # Resize to face crop size
            heatmap_resized = cv2.resize(heatmap, (face_crop.shape[1], face_crop.shape[0]))
            
            return {
                'gradcam_heatmap': heatmap_resized,
                'target_class': target_class,
                'target_emotion': self.emotion_analyzer.model.config.id2label[target_class].lower(),
                'face_region': face_region,
                'face_crop': face_crop
            }
        
        except Exception as e:
            print(f"[EmotionProcessor] Grad-CAM extraction error: {e}")
            return None
    
    def create_xai_overlay(self, frame: np.ndarray, xai_data: Dict, method: str = 'attention') -> np.ndarray:
        """
        Create visualization overlay of attention map or Grad-CAM on original frame.
        
        Args:
            frame: Original frame
            xai_data: XAI data dictionary
            method: 'attention' or 'gradcam'
            
        Returns:
            Frame with XAI overlay
        """
        overlay = frame.copy()
        
        if xai_data is None:
            return overlay
        
        try:
            face_region = xai_data['face_region']
            face_crop = xai_data['face_crop']
            
            # Get heatmap based on method
            if method == 'attention':
                heatmap = xai_data['attention_map']
            else:  # gradcam
                heatmap = xai_data['gradcam_heatmap']
            
            # Convert heatmap to color
            heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Blend with face crop
            alpha = 0.4
            blended_face = cv2.addWeighted(face_crop, 1-alpha, heatmap_color, alpha, 0)
            
            # Place blended face back on frame
            x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
            overlay[y:y+h, x:x+w] = blended_face
            
            # Draw face bounding box
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # Add label
            label = f"XAI: {method.upper()}"
            cv2.putText(overlay, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        except Exception as e:
            print(f"[EmotionProcessor] Overlay creation error: {e}")
        
        return overlay
    
    def _calculate_xai_target_frames(self, total_frames: int, timeline: List[Dict]) -> List[int]:
        """
        Calculate 4 strategic frame indices for XAI extraction.
        
        Strategy:
        - Frame 1: ~25% of video (early baseline)
        - Frame 2: ~50% of video (midpoint)
        - Frame 3: ~75% of video (later analysis)
        - Frame 4: Frame with highest emotion confidence (peak moment)
        
        Args:
            total_frames: Total number of frames in video
            timeline: Processed timeline with emotion data
            
        Returns:
            List of 4 frame indices
        """
        if total_frames < 4:
            return list(range(total_frames))
        
        # Calculate percentage-based frames
        frame_25 = int(total_frames * 0.25)
        frame_50 = int(total_frames * 0.50)
        frame_75 = int(total_frames * 0.75)
        
        # Find frame with peak emotion (highest confidence in dominant emotion)
        peak_frame = frame_50  # Default to midpoint
        peak_confidence = 0.0
        
        for entry in timeline:
            if entry['success'] and entry.get('emotions'):
                dominant = entry.get('dominant_emotion')
                if dominant and dominant in entry['emotions']:
                    confidence = entry['emotions'][dominant]
                    if confidence > peak_confidence:
                        peak_confidence = confidence
                        peak_frame = entry['frame']
        
        # Build list of target frames, avoiding duplicates
        target_frames = []
        candidates = [frame_25, frame_50, frame_75, peak_frame]
        
        for frame_idx in candidates:
            # Find nearest successful frame
            best_frame = self._find_nearest_successful_frame(frame_idx, timeline, target_frames)
            if best_frame is not None and best_frame not in target_frames:
                target_frames.append(best_frame)
        
        # Ensure we have exactly 4 frames (fill with other successful frames if needed)
        if len(target_frames) < 4:
            for entry in timeline:
                if entry['success'] and entry['frame'] not in target_frames:
                    target_frames.append(entry['frame'])
                    if len(target_frames) >= 4:
                        break
        
        return sorted(target_frames[:4])
    
    def _find_nearest_successful_frame(self, target_idx: int, timeline: List[Dict], exclude: List[int]) -> Optional[int]:
        """
        Find the nearest frame with successful emotion detection.
        
        Args:
            target_idx: Target frame index
            timeline: Processed timeline
            exclude: Frame indices to exclude
            
        Returns:
            Nearest successful frame index or None
        """
        if target_idx < 0 or target_idx >= len(timeline):
            return None
        
        # Check target frame first
        if timeline[target_idx]['success'] and target_idx not in exclude:
            return target_idx
        
        # Search outward from target
        max_search = len(timeline)
        for offset in range(1, max_search):
            # Check frame before
            if target_idx - offset >= 0:
                if timeline[target_idx - offset]['success'] and (target_idx - offset) not in exclude:
                    return target_idx - offset
            # Check frame after
            if target_idx + offset < len(timeline):
                if timeline[target_idx + offset]['success'] and (target_idx + offset) not in exclude:
                    return target_idx + offset
        
        return None
    
    def _encode_face_image_base64(self, frame: np.ndarray) -> Optional[str]:
        """
        Detect face, crop it, and encode as Base64 JPEG.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Base64 encoded face image string or None
        """
        try:
            face_crop, face_region = self.emotion_analyzer._detect_and_crop_face(frame)
            if face_crop is None:
                return None
            
            # Resize face to reasonable size for storage (150x150)
            face_resized = cv2.resize(face_crop, (150, 150))
            
            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', face_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Convert to Base64
            face_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return face_base64
        except Exception as e:
            print(f"[EmotionProcessor] Face encoding error: {e}")
            return None

    def compute_emotion_summary(self, frames: List[np.ndarray], fps: float, extract_xai: bool = True) -> Dict:
        """
        Process frames through emotion analyzer and generate timeline + summary.
        
        Args:
            frames: List of video frames (BGR format)
            fps: Frames per second
            extract_xai: Whether to extract XAI data (attention maps, Grad-CAM)
            
        Returns:
            Dictionary with:
            - timeline: List of per-frame emotion data
            - summary: Aggregate emotion metrics
            - xai_frames: Frames with XAI data available (if extract_xai=True)
        """
        print(f"[EmotionProcessor] Processing {len(frames)} frames for emotion analysis...")
        if extract_xai:
            print(f"[EmotionProcessor] XAI extraction enabled (4 strategic frames)")
        
        # Reset analyzer
        self.emotion_analyzer.emotion_counts = {e: 0 for e in self.emotion_analyzer.emotions}
        self.emotion_analyzer.all_frames_emotions = []
        
        timeline = []
        xai_frames = []  # Frame indices where XAI data is available
        
        # PHASE 1: Process all frames for emotion detection
        print(f"[EmotionProcessor] Phase 1: Detecting emotions in all frames...")
        for frame_idx, frame in enumerate(frames):
            # Calculate timestamp
            timestamp = frame_idx / fps if fps > 0 else 0
            
            # Detect emotion
            emotion_data = self.emotion_analyzer.detect_emotion(frame)
            
            # Build timeline entry
            timeline_entry = {
                't': round(timestamp, 3),
                'frame': frame_idx,
                'success': emotion_data['success'],
                'emotions': emotion_data.get('emotions', {}),
                'dominant_emotion': emotion_data.get('dominant_emotion'),
                'dominant_emotion_code': emotion_data.get('dominant_emotion_code'),
                'face_region': emotion_data.get('face_region'),
                'has_xai': False
            }
            
            # Track emotions
            if emotion_data['success']:
                dominant = emotion_data['dominant_emotion']
                self.emotion_analyzer.emotion_counts[dominant] += 1
                self.emotion_analyzer.all_frames_emotions.append(dominant)
            
            timeline.append(timeline_entry)
        
        # PHASE 2: Extract XAI for 4 strategic frames
        if extract_xai:
            print(f"[EmotionProcessor] Phase 2: Selecting 4 strategic frames for XAI...")
            target_frames = self._calculate_xai_target_frames(len(frames), timeline)
            print(f"[EmotionProcessor] Target XAI frames: {target_frames}")
            
            for frame_idx in target_frames:
                if frame_idx >= len(frames):
                    continue
                    
                frame = frames[frame_idx]
                entry = timeline[frame_idx]
                
                if not entry['success']:
                    continue
                
                print(f"[EmotionProcessor] Extracting XAI for frame {frame_idx} (t={entry['t']}s, {entry['dominant_emotion']})...")
                
                # Extract attention maps
                attention_data = self.extract_attention_maps(frame)
                if attention_data:
                    entry['attention_map'] = attention_data['attention_map'].tolist()
                    entry['attention_grid_size'] = attention_data['grid_size']
                
                # Extract Grad-CAM
                gradcam_data = self.extract_gradcam(frame, target_emotion=entry.get('dominant_emotion'))
                if gradcam_data:
                    entry['gradcam_heatmap'] = gradcam_data['gradcam_heatmap'].tolist()
                    entry['gradcam_target'] = gradcam_data['target_emotion']
                
                # Encode face image for visualization
                face_base64 = self._encode_face_image_base64(frame)
                if face_base64:
                    entry['face_image_base64'] = face_base64
                
                if attention_data or gradcam_data:
                    entry['has_xai'] = True
                    xai_frames.append(frame_idx)
        
        # Calculate summary statistics
        total_frames = len(frames)
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        # Count successful detections
        successful_detections = sum(1 for entry in timeline if entry['success'])
        detection_rate = (successful_detections / total_frames * 100) if total_frames > 0 else 0
        
        # Determine dominant emotion
        if self.emotion_analyzer.all_frames_emotions:
            emotion_counter = Counter(self.emotion_analyzer.all_frames_emotions)
            dominant_emotion = emotion_counter.most_common(1)[0][0]
            dominant_emotion_code = self.emotion_analyzer.emotion_map.get(dominant_emotion, 0)
        else:
            dominant_emotion = 'neutral'
            dominant_emotion_code = 0
        
        # Build emotion distribution
        distribution = self.emotion_analyzer.emotion_counts.copy()
        
        # Build summary
        summary = {
            'total_frames': total_frames,
            'duration_seconds': round(duration_seconds, 2),
            'successful_detections': successful_detections,
            'detection_rate': round(detection_rate, 2),
            'dominant_emotion': dominant_emotion,
            'dominant_emotion_code': dominant_emotion_code,
            'distribution': distribution,
            'xai_available': extract_xai,
            'xai_frame_count': len(xai_frames),
            'xai_frames': xai_frames
        }
        
        print(f"[EmotionProcessor] ✓ Processing complete")
        print(f"[EmotionProcessor]   Dominant emotion: {dominant_emotion}")
        print(f"[EmotionProcessor]   Detection rate: {detection_rate:.1f}%")
        if extract_xai:
            print(f"[EmotionProcessor]   XAI data extracted for {len(xai_frames)} frames: {xai_frames}")
        
        return {
            'timeline': timeline,
            'summary': summary
        }
    
    def reset(self):
        """Reset the processor to clean state."""
        self.emotion_analyzer.emotion_counts = {e: 0 for e in self.emotion_analyzer.emotions}
        self.emotion_analyzer.all_frames_emotions = []
        self.emotion_analyzer.last_successful_result = None
        self.emotion_analyzer.consecutive_failures = 0
        self.emotion_analyzer.result_cache = {}
