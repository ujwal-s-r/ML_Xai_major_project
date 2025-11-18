import cv2
import numpy as np
from collections import Counter
import os
import time
from typing import Dict, List, Tuple, Union, Optional
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import mediapipe as mp

class EmotionAnalyzer:
    def __init__(self, processor: Optional[AutoImageProcessor] = None, model: Optional[AutoModelForImageClassification] = None, device: Optional[str] = None, enable_face_detector: bool = True):
        """Initialize the emotion analyzer with a ViT-based model (Transformers).

        Optional args allow injecting mocks in tests or custom devices.
        """
        # Supported emotions (aligned with model labels)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # Emotion map for integer coding
        self.emotion_map = {
            'neutral': 0,
            'sad': 1,
            'happy': 2,
            'angry': 3,
            'fear': 4,
            'surprise': 5,
            'disgust': 6
        }

        # Load ViT model and processor
        model_name = "dima806/facial_emotions_image_detection"
        if processor is None or model is None:
            print(f"[EmotionAnalyzer] Loading model: {model_name}")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
        else:
            self.processor = processor
            self.model = model

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"[EmotionAnalyzer] Model loaded on device: {self.device}")

        # Initialize MediaPipe face detection to crop faces before classification
        self.face_detection = None
        if enable_face_detector:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )

        # Error handling state
        self.last_successful_result = None
        self.consecutive_failures = 0
        self.max_failures = 5

        # Results storage
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}
        self.all_frames_emotions = []
        self.dominant_emotion = None
        self.dominant_emotion_code = None

        # Cache for results to avoid redundant processing
        self.result_cache = {}
        self.cache_max_size = 30  # Limit cache size to prevent memory issues
        
    def _detect_and_crop_face(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Detect face using MediaPipe and crop ROI for classification.

        Returns (face_crop_bgr, face_region_dict) or (None, None) if no face.
        """
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            results = self.face_detection.process(rgb)
            if not results.detections:
                return None, None

            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Pad box a bit
            pad = 20
            x = max(0, x - pad)
            y = max(0, y - pad)
            bw = min(w - x, bw + 2 * pad)
            bh = min(h - y, bh + 2 * pad)

            crop = frame[y:y+bh, x:x+bw]
            region = {"x": x, "y": y, "w": bw, "h": bh}
            return crop, region
        except Exception as e:
            print(f"[EmotionAnalyzer] Face detection error: {e}")
            return None, None
    
    def detect_emotion(self, frame: np.ndarray) -> Dict:
        """Detect emotions in a single frame using ViT model."""
        # Initialize return data
        emotion_data = {
            'emotions': None,
            'dominant_emotion': None,
            'dominant_emotion_code': None,
            'success': False,
            'face_region': None
        }

        # Validate frame
        if frame is None or frame.size == 0:
            print("[EmotionAnalyzer] Invalid frame provided to detect_emotion")
            if self.last_successful_result:
                result = self.last_successful_result.copy()
                result['fresh'] = False
                return result
            return emotion_data

        # Cache key for identical frames
        try:
            small = cv2.resize(frame, (100, 100))
            cache_key = hash(small.tobytes())
            if cache_key in self.result_cache:
                return self.result_cache[cache_key]
        except Exception as e:
            print(f"[EmotionAnalyzer] Cache key error: {e}")
            cache_key = None

        try:
            # Detect and crop face
            face_crop, face_region = self._detect_and_crop_face(frame)
            if face_crop is None:
                self.consecutive_failures += 1
                if self.last_successful_result:
                    result = self.last_successful_result.copy()
                    result['fresh'] = False
                    return result
                return emotion_data

            # Convert to PIL and preprocess
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_face)
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # Map id->label and build percentage dict
            id2label = self.model.config.id2label
            emotions_dict = {lbl.lower(): 0.0 for lbl in self.emotions}
            for idx, p in enumerate(probs):
                lbl = id2label[idx].lower()
                emotions_dict[lbl] = float(p * 100.0)

            # Determine dominant emotion
            dominant_emotion = max(emotions_dict, key=emotions_dict.get)
            dominant_emotion_code = self.emotion_map.get(dominant_emotion, -1)

            # Success
            self.consecutive_failures = 0
            emotion_data['emotions'] = emotions_dict
            emotion_data['dominant_emotion'] = dominant_emotion
            emotion_data['dominant_emotion_code'] = dominant_emotion_code
            emotion_data['success'] = True
            emotion_data['face_region'] = face_region
            emotion_data['fresh'] = True

            # Cache
            if cache_key is not None:
                self.result_cache[cache_key] = emotion_data.copy()
                if len(self.result_cache) > self.cache_max_size:
                    self.result_cache.pop(next(iter(self.result_cache)))

            # Track last successful
            self.last_successful_result = emotion_data.copy()

        except Exception as e:
            print(f"[EmotionAnalyzer] ViT detection error: {e}")
            self.consecutive_failures += 1
            if self.last_successful_result:
                result = self.last_successful_result.copy()
                result['fresh'] = False
                return result

        return emotion_data

    def process_video(self, video_path: str, output_path: Optional[str] = None, fps: int = 10) -> int:
        """
        Process a video file to detect emotions at the specified FPS.
        
        Args:
            video_path: Path to the video file
            output_path: Optional path to save processed video
            fps: Frames per second to process (default 10)
            
        Returns:
            The dominant emotion code across the entire video
        """
        print(f"Starting video processing: {video_path}")
        
        # Reset counts
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}
        self.all_frames_emotions = []
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return 0  # Return neutral as default
            
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(video_fps / fps))  # Process at specified FPS, minimum 1
        
        # Initialize writer if output is specified
        writer = None
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        
        print(f"Video properties: {video_fps} FPS, {total_frames} frames")
        print(f"Processing at {fps} FPS (every {frame_interval} frames)")
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Process only every n-th frame to achieve target FPS
                if frame_count % frame_interval != 0:
                    continue
                    
                processed_count += 1
                
                # Detect emotion in the frame
                emotion_data = self.detect_emotion(frame)
                
                if emotion_data['success']:
                    # Track the dominant emotion
                    dominant_emotion = emotion_data['dominant_emotion']
                    self.emotion_counts[dominant_emotion] += 1
                    self.all_frames_emotions.append(dominant_emotion)
                    
                    # Update visualization if needed
                    if writer:
                        vis_frame = self.visualize_emotion_detection(frame, emotion_data)
                        writer.write(vis_frame)
                
                # Print progress periodically
                if processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% - Processed {processed_count} frames in {elapsed:.1f}s")
        
        except Exception as e:
            print(f"Error during video processing: {e}")
        
        finally:
            # Clean up
            cap.release()
            if writer:
                writer.release()

        # Calculate the dominant emotion across all frames
        if self.all_frames_emotions:
            most_common = Counter(self.all_frames_emotions).most_common(1)[0][0]
            self.dominant_emotion = most_common
            self.dominant_emotion_code = self.emotion_map[most_common]

            print(f"Analysis complete. Dominant emotion: {self.dominant_emotion} (code: {self.dominant_emotion_code})")
            print(f"Emotion distribution: {self.emotion_counts}")
        else:
            print("No emotions detected in the video")
            self.dominant_emotion = "neutral"
            self.dominant_emotion_code = 0

        return self.dominant_emotion_code
        
    def visualize_emotion_detection(self, frame: np.ndarray, emotion_data: Dict) -> np.ndarray:
        """
        Visualize emotion detection results on a frame.
        
        Args:
            frame: Input frame from video
            emotion_data: Dictionary from detect_emotion
            
        Returns:
            Frame with emotion visualization
        """
        output_frame = frame.copy()
        
        if emotion_data['success']:
            # Get emotion data
            emotions = emotion_data['emotions']
            dominant_emotion = emotion_data['dominant_emotion']
            
            # Draw emotion label with freshness indicator
            freshness_indicator = ""
            if 'fresh' in emotion_data and not emotion_data.get('fresh', True):
                freshness_indicator = " (cached)"
                
            cv2.putText(
                output_frame,
                f"Emotion: {dominant_emotion}{freshness_indicator}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Draw emotion probabilities
            y_offset = 60
            for emotion, probability in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                # Format probability as percentage
                text = f"{emotion}: {probability:.1f}%"
                
                cv2.putText(
                    output_frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
                y_offset += 25
                
            # Try to draw face region if available
            face_region = emotion_data.get('face_region', None)
            if face_region:
                try:
                    x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error drawing face region: {e}")
        else:
            # No emotion detected
            error_msg = "No face detected"
            if 'error' in emotion_data:
                error_msg = f"Error: {emotion_data['error']}"
                
            cv2.putText(
                output_frame,
                error_msg,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        return output_frame 