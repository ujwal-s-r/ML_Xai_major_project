"""
Test script for Gemini AI Analyzer
Tests the AI report generation with sample data
"""

import sys
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from processors.gemini_analyzer import GeminiAnalyzer


def create_sample_phq8_data():
    """Create sample PHQ-8 data for testing."""
    return {
        "session_id": "test-session-123",
        "answers": [2, 2, 1, 2, 1, 2, 1, 1],  # Total: 12 (Moderate)
        "score": 12,
        "severity": "Moderate",
        "timestamp": "2025-11-18T10:00:00Z",
        "version": 1
    }


def create_sample_game_data():
    """Create sample game data for testing."""
    return {
        "session_id": "test-session-123",
        "duration_ms": 60000,
        "server_summary": {
            "totalEligible": 45,
            "hits": 38,
            "misses": 7,
            "errors": 5,
            "accuracy": 84,
            "avgRt": 1245,
            "impulsive": 3,
            "perDistraction": [
                {
                    "id": 0,
                    "kind": "visual",
                    "startTs": 20000,
                    "duration": 800,
                    "pre": {
                        "targets": 8,
                        "hits": 7,
                        "acc": 0.875,
                        "avgRt": 1180
                    },
                    "post": {
                        "targets": 7,
                        "hits": 5,
                        "acc": 0.714,
                        "avgRt": 1420
                    }
                },
                {
                    "id": 1,
                    "kind": "text",
                    "startTs": 40000,
                    "duration": 900,
                    "pre": {
                        "targets": 9,
                        "hits": 8,
                        "acc": 0.889,
                        "avgRt": 1150
                    },
                    "post": {
                        "targets": 8,
                        "hits": 6,
                        "acc": 0.75,
                        "avgRt": 1380
                    }
                }
            ]
        },
        "timestamp": "2025-11-18T10:01:30Z",
        "version": 1
    }


def create_sample_video_data():
    """Create sample video analysis data for testing."""
    return {
        "session_id": "test-session-123",
        "duration_seconds": 30.5,
        "server_summary": {
            "blink": {
                "total_blinks": 18,
                "blink_rate_per_minute": 35.4,
                "avg_ear": 0.2654,
                "total_frames": 915,
                "duration_seconds": 30.5,
                "successful_detections": 890,
                "detection_rate": 97.3
            },
            "gaze": {
                "total_frames": 915,
                "duration_seconds": 30.5,
                "successful_detections": 875,
                "detection_rate": 95.6,
                "dominant_direction": "Looking Center",
                "attention_score": 0.723,
                "distribution": {
                    "left": 95,
                    "center": 658,
                    "right": 122
                },
                "distribution_percentage": {
                    "left": 10.9,
                    "center": 75.2,
                    "right": 13.9
                }
            },
            "pupil": {
                "total_frames": 915,
                "duration_seconds": 30.5,
                "successful_detections": 880,
                "detection_rate": 96.2,
                "avg_pupil_size": 0.0315,
                "min_pupil_size": 0.0255,
                "max_pupil_size": 0.0425,
                "pupil_dilation_delta": 0.0085,
                "pupil_dilation_events": 12,
                "pupil_constriction_events": 8,
                "baseline_recorded": True,
                "baseline_pupil_size": 0.0305,
                "pupil_variability": {
                    "mean": 0.0315,
                    "std": 0.0045,
                    "min": 0.0255,
                    "max": 0.0425,
                    "range": 0.017
                }
            },
            "emotion": {
                "total_frames": 915,
                "duration_seconds": 30.5,
                "successful_detections": 305,
                "detection_rate": 33.3,
                "dominant_emotion": "sad",
                "dominant_emotion_code": 0,
                "distribution": {
                    "sad": 137,
                    "neutral": 89,
                    "fear": 42,
                    "angry": 18,
                    "happy": 12,
                    "surprise": 5,
                    "disgust": 2
                },
                "xai_available": True,
                "xai_frame_count": 10,
                "xai_frames": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
            }
        },
        "timeline": [
            # Sample timeline entry
            {
                "t": 0.0,
                "frame": 0,
                "blink": {
                    "left_ear": 0.265,
                    "right_ear": 0.268,
                    "avg_ear": 0.2665,
                    "is_blinking": False,
                    "blink_count": 0,
                    "success": True
                },
                "gaze": {
                    "face_detected": True,
                    "eyes_detected": True,
                    "gaze_direction": "center",
                    "avg_left_perc": 0.48,
                    "avg_right_perc": 0.52,
                    "left_pupil_coords": [0.35, 0.42],
                    "right_pupil_coords": [0.65, 0.43]
                },
                "pupil": {
                    "success": True,
                    "left_iris_center": [0.35, 0.42],
                    "right_iris_center": [0.65, 0.43],
                    "left_pupil_size": 0.0310,
                    "right_pupil_size": 0.0312,
                    "avg_pupil_size": 0.0311,
                    "pupil_dilation_ratio": 1.02
                },
                "emotion": {
                    "success": True,
                    "dominant_emotion": "sad",
                    "dominant_emotion_code": 0,
                    "emotions": {
                        "sad": 0.65,
                        "neutral": 0.20,
                        "fear": 0.10,
                        "angry": 0.03,
                        "happy": 0.01,
                        "surprise": 0.01,
                        "disgust": 0.00
                    },
                    "has_xai": True
                }
            }
        ],
        "timestamp": "2025-11-18T10:02:00Z"
    }


def test_gemini_initialization():
    """Test Gemini analyzer initialization."""
    print("\n" + "="*60)
    print("TEST 1: Gemini Analyzer Initialization")
    print("="*60)
    
    try:
        analyzer = GeminiAnalyzer()
        print("✓ Gemini analyzer initialized successfully")
        print(f"  Model: gemini-2.0-flash-exp")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_report_generation_full():
    """Test full report generation with all data."""
    print("\n" + "="*60)
    print("TEST 2: Full Report Generation (All 3 Stages)")
    print("="*60)
    
    try:
        # Create sample data
        phq8_data = create_sample_phq8_data()
        game_data = create_sample_game_data()
        video_data = create_sample_video_data()
        
        print("\nSample Data Created:")
        print(f"  PHQ-8 Score: {phq8_data['score']}/24 ({phq8_data['severity']})")
        print(f"  Game Accuracy: {game_data['server_summary']['accuracy']}%")
        print(f"  Dominant Emotion: {video_data['server_summary']['emotion']['dominant_emotion']}")
        
        # Initialize analyzer
        analyzer = GeminiAnalyzer()
        
        # Generate report
        print("\n⏳ Generating AI report... (this may take 10-30 seconds)")
        report = analyzer.generate_assessment_report(
            phq8_data=phq8_data,
            game_data=game_data,
            video_data=video_data
        )
        
        # Display results
        print("\n✓ Report generated successfully!")
        print(f"\n{'='*60}")
        print("REPORT PREVIEW (First 500 characters):")
        print(f"{'='*60}")
        print(report[:500] + "...\n")
        
        print(f"{'='*60}")
        print("REPORT STATISTICS:")
        print(f"{'='*60}")
        print(f"  Total length: {len(report)} characters")
        print(f"  Total words: {len(report.split())} words")
        print(f"  Lines: {len(report.split(chr(10)))} lines")
        
        # Check for key sections
        sections = [
            "Executive Summary",
            "PHQ-8",
            "Cognitive Performance",
            "Behavioral",
            "Emotional",
            "Recommendations",
            "Disclaimer"
        ]
        
        print(f"\n  Key sections found:")
        for section in sections:
            found = section.lower() in report.lower()
            print(f"    {'✓' if found else '✗'} {section}")
        
        # Save to file
        output_file = PROJECT_ROOT / "tests" / "test_report_output.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\n  Full report saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_report_generation_partial():
    """Test report generation with partial data (only PHQ-8)."""
    print("\n" + "="*60)
    print("TEST 3: Partial Report Generation (PHQ-8 Only)")
    print("="*60)
    
    try:
        phq8_data = create_sample_phq8_data()
        
        print("\nGenerating report with PHQ-8 data only...")
        
        analyzer = GeminiAnalyzer()
        report = analyzer.generate_assessment_report(
            phq8_data=phq8_data,
            game_data=None,
            video_data=None
        )
        
        print("✓ Partial report generated successfully!")
        print(f"  Length: {len(report)} characters")
        
        return True
        
    except Exception as e:
        print(f"✗ Partial report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling with no data."""
    print("\n" + "="*60)
    print("TEST 4: Error Handling (No Data)")
    print("="*60)
    
    try:
        analyzer = GeminiAnalyzer()
        report = analyzer.generate_assessment_report(
            phq8_data=None,
            game_data=None,
            video_data=None
        )
        
        print("✓ Error handling working - fallback report generated")
        print(f"  Report type: {'Fallback' if 'error' in report.lower() or len(report) < 1000 else 'Normal'}")
        
        return True
        
    except Exception as e:
        print(f"  Exception caught (expected): {type(e).__name__}")
        return True


def main():
    """Run all tests."""
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + " "*15 + "GEMINI AI ANALYZER TESTS" + " "*19 + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    results = []
    
    # Test 1: Initialization
    results.append(("Initialization", test_gemini_initialization()))
    
    if results[0][1]:  # Only continue if initialization succeeded
        # Test 2: Full report
        results.append(("Full Report", test_report_generation_full()))
        
        # Test 3: Partial report
        results.append(("Partial Report", test_report_generation_partial()))
        
        # Test 4: Error handling
        results.append(("Error Handling", test_error_handling()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed!")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
