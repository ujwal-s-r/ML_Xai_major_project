"""
Gemini AI Analyzer
Generates comprehensive mental health assessment reports using Google's Gemini 2.0 Flash model.
Analyzes PHQ-8, game performance, and video analysis data to provide interpretable insights.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)


class GeminiAnalyzer:
    """
    AI-powered mental health assessment analyzer using Gemini 2.0 Flash.
    Generates comprehensive, human-readable reports from multi-stage assessment data.
    """
    
    def __init__(self):
        """Initialize Gemini AI with API key."""
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please add it to your .env file in the project root."
            )
        
        genai.configure(api_key=api_key)
        
        # Use Gemini 2.0 Flash model
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        print("[GeminiAnalyzer] Initialized with gemini-2.0-flash-exp model")
    
    def generate_assessment_report(
        self,
        phq8_data: Optional[Dict] = None,
        game_data: Optional[Dict] = None,
        video_data: Optional[Dict] = None
    ) -> str:
        """
        Generate comprehensive mental health assessment report.
        
        Args:
            phq8_data: PHQ-8 questionnaire results
            game_data: Buzzer Bombardment game performance data
            video_data: Video analysis results (emotion, blink, gaze, pupil)
            
        Returns:
            Markdown-formatted assessment report
        """
        prompt = self._build_comprehensive_prompt(phq8_data, game_data, video_data)
        
        print("[GeminiAnalyzer] Generating AI assessment report...")
        print(f"[GeminiAnalyzer] Prompt length: {len(prompt)} characters")
        
        try:
            response = self.model.generate_content(prompt)
            report = response.text
            
            print(f"[GeminiAnalyzer] ✓ Report generated ({len(report)} characters)")
            return report
            
        except Exception as e:
            print(f"[GeminiAnalyzer] Error generating report: {e}")
            return self._generate_fallback_report(e)
    
    def _build_comprehensive_prompt(
        self,
        phq8_data: Optional[Dict],
        game_data: Optional[Dict],
        video_data: Optional[Dict]
    ) -> str:
        """
        Build detailed prompt with all assessment data and context.
        """
        prompt_parts = []
        
        # System context
        prompt_parts.append("""# Mental Health Assessment Analysis Request

You are an expert clinical psychologist and mental health assessment specialist. You have been provided with comprehensive data from a multi-stage mental health assessment system that combines:

1. **PHQ-8 Depression Screening** - A validated questionnaire for depression severity
2. **Buzzer Bombardment Cognitive Game** - A reaction time test measuring attention, impulsivity, and cognitive performance under distraction
3. **Real-time Video Analysis** - Advanced ML-based facial analysis capturing emotional responses, blink patterns, pupil dilation, and gaze behavior

Your task is to analyze this data holistically and generate a comprehensive, compassionate, and actionable assessment report.

---

## ASSESSMENT DATA

""")
        
        # PHQ-8 Section
        if phq8_data:
            prompt_parts.append(self._format_phq8_section(phq8_data))
        
        # Game Section
        if game_data:
            prompt_parts.append(self._format_game_section(game_data))
        
        # Video Analysis Section
        if video_data:
            prompt_parts.append(self._format_video_section(video_data))
        
        # Instructions for report generation
        prompt_parts.append("""

---

## REPORT GENERATION INSTRUCTIONS

Please generate a comprehensive mental health assessment report in **Markdown format** with the following structure:

### 1. Executive Summary
- Brief overview of all three assessment stages
- Overall severity level and key concerns
- Primary recommendations (2-3 sentences)

### 2. PHQ-8 Depression Assessment
- Score interpretation (0-24 scale)
- Severity level analysis (None/Mild/Moderate/Moderately Severe/Severe)
- Question-by-question insights highlighting concerning patterns
- Clinical significance of the responses

### 3. Cognitive Performance Analysis (Buzzer Bombardment Game)
- Overall performance metrics (accuracy, reaction time)
- Attention and focus capacity
- Impulsivity indicators and their significance
- Impact of distractions on performance (pre vs post comparison)
- Rule-following ability under pressure
- Cognitive flexibility assessment

### 4. Behavioral & Emotional Analysis (Video Analysis)
- **Emotional Patterns**: Dominant emotions observed, emotional variability, stress indicators
- **Blink Analysis**: Blink rate interpretation (normal: 15-20/min), stress/anxiety indicators
- **Pupil Response**: Dilation patterns, autonomic nervous system indicators, stress responses
- **Gaze Behavior**: Attention patterns, avoidance behaviors, engagement levels
- Integration of all four metrics for comprehensive behavioral assessment

### 5. Integrated Clinical Interpretation
- How do the three stages corroborate or contradict each other?
- What patterns emerge across psychological, cognitive, and behavioral domains?
- Strengths and areas of concern
- Potential underlying mechanisms

### 6. Recommendations
- **Immediate Actions**: What should be done right away
- **Professional Support**: Suggested type of mental health support (if needed)
- **Self-Care Strategies**: Evidence-based coping mechanisms
- **Follow-up**: Suggested timeline for reassessment

### 7. Important Disclaimers
- This is a screening tool, not a diagnosis
- Importance of professional consultation
- Crisis resources if needed

---

## IMPORTANT GUIDELINES

1. **Be Compassionate**: Use empathetic, non-judgmental language
2. **Be Specific**: Reference actual data points and metrics
3. **Be Actionable**: Provide concrete, practical recommendations
4. **Be Accurate**: Base interpretations on clinical research and validated norms
5. **Be Balanced**: Acknowledge strengths as well as concerns
6. **Use Markdown**: Proper headings (##, ###), lists, **bold**, *italic*, and tables where appropriate
7. **Length**: Aim for comprehensive coverage (1500-2500 words)
8. **Tone**: Professional yet accessible, suitable for both patients and clinicians

---

Please generate the assessment report now:
""")
        
        return "\n".join(prompt_parts)
    
    def _format_phq8_section(self, phq8_data: Dict) -> str:
        """Format PHQ-8 data with full context."""
        
        # PHQ-8 questions for reference
        questions = [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
            "Trouble concentrating on things, such as reading the newspaper or watching television",
            "Moving or speaking so slowly that other people could have noticed. Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual"
        ]
        
        score_labels = ["Not at all (0)", "Several days (1)", "More than half the days (2)", "Nearly every day (3)"]
        
        section = ["\n### 1. PHQ-8 DEPRESSION SCREENING QUESTIONNAIRE\n"]
        section.append("**Context**: The PHQ-8 is a validated 8-item depression screening tool derived from the PHQ-9. Each question assesses symptoms over the past 2 weeks on a 0-3 scale.\n")
        section.append("**Scoring**: 0-4 (None), 5-9 (Mild), 10-14 (Moderate), 15-19 (Moderately Severe), 20-24 (Severe)\n")
        
        answers = phq8_data.get('answers', [])
        total_score = phq8_data.get('score', sum(answers))
        severity = phq8_data.get('severity', 'Unknown')
        
        section.append(f"\n**Total Score**: {total_score}/24")
        section.append(f"**Severity Level**: {severity}\n")
        
        section.append("\n**Individual Question Responses**:\n")
        for i, (question, answer) in enumerate(zip(questions, answers), 1):
            section.append(f"{i}. {question}")
            section.append(f"   - Response: **{score_labels[answer]}** (Score: {answer})\n")
        
        return "\n".join(section)
    
    def _format_game_section(self, game_data: Dict) -> str:
        """Format game performance data with full context."""
        
        section = ["\n### 2. BUZZER BOMBARDMENT COGNITIVE GAME\n"]
        
        section.append("""**Game Description**: 
The Buzzer Bombardment is a 60-second reaction time and attention assessment game designed to measure:
- **Selective Attention**: Ability to tap correct targets (circles, squares, triangles) while ignoring distractors
- **Sustained Attention**: Maintaining focus throughout the 60-second duration
- **Inhibitory Control**: Avoiding fake/red targets and resisting impulsive responses
- **Cognitive Flexibility**: Adapting to dynamic rule changes (e.g., "only tap squares")
- **Distraction Resilience**: Performance impact when visual/auditory distractions occur

**Game Mechanics**:
- Duration: 60 seconds
- Targets: Green shapes (circles, squares, triangles) that must be tapped
- Fake Targets: Red shapes that should NOT be tapped
- Distractions: Visual overlays and rule change announcements at 20s and 40s
- Adaptive Difficulty: Game adjusts spawn rate and fake probability based on performance
- Rule Windows: Temporary periods where only specific shapes are valid (e.g., "only squares")

**Metrics Explained**:
- **Accuracy**: Percentage of valid targets successfully tapped
- **Reaction Time (RT)**: Average time from target appearance to tap (milliseconds)
- **Impulsivity**: Count of premature taps (before target appears) + taps on fake targets
- **Errors**: Total incorrect responses (fake taps, wrong shapes during rules, premature taps)
- **Pre/Post Distraction**: Performance comparison in 5-second windows before and after each distraction

""")
        
        # Get server summary with safe fallbacks
        summary = game_data.get('server_summary', game_data.get('summary', {}))
        
        accuracy = summary.get('accuracy', 0)
        avg_rt = summary.get('avgRt', 0)
        impulsive = summary.get('impulsive', 0)
        errors = summary.get('errors', 0)
        total_eligible = summary.get('totalEligible', 0)
        hits = summary.get('hits', 0)
        misses = summary.get('misses', 0)
        
        section.append("\n**Overall Performance Metrics**:\n")
        section.append(f"- **Accuracy**: {accuracy}% ({hits}/{total_eligible} valid targets hit)")
        section.append(f"- **Misses**: {misses} valid targets missed")
        section.append(f"- **Average Reaction Time**: {avg_rt}ms" if avg_rt else "- **Average Reaction Time**: N/A")
        section.append(f"- **Impulsivity Score**: {impulsive} (premature taps + fake target taps)")
        section.append(f"- **Total Errors**: {errors}\n")
        
        # Distraction impact analysis
        per_distraction = summary.get('perDistraction', [])
        if per_distraction:
            section.append("\n**Distraction Impact Analysis**:\n")
            section.append("The game includes 2 planned distractions to assess resilience under stress:\n")
            
            for dist in per_distraction:
                dist_num = dist.get('id', 0) + 1
                kind = dist.get('kind', 'unknown')
                start_time = dist.get('startTs', 0) / 1000  # Convert to seconds
                
                pre = dist.get('pre', {})
                post = dist.get('post', {})
                
                pre_acc = pre.get('acc')
                post_acc = post.get('acc')
                pre_rt = pre.get('avgRt')
                post_rt = post.get('avgRt')
                pre_targets = pre.get('targets', 0)
                post_targets = post.get('targets', 0)
                pre_hits = pre.get('hits', 0)
                post_hits = post.get('hits', 0)
                
                section.append(f"\n**Distraction #{dist_num}** ({kind} at {start_time:.1f}s):")
                
                if pre_acc is not None and post_acc is not None:
                    acc_change = (post_acc - pre_acc) * 100
                    section.append(f"- Pre-distraction accuracy: {pre_acc*100:.1f}% ({pre_hits}/{pre_targets})")
                    section.append(f"- Post-distraction accuracy: {post_acc*100:.1f}% ({post_hits}/{post_targets})")
                    section.append(f"- Accuracy change: {acc_change:+.1f}%")
                
                if pre_rt is not None and post_rt is not None:
                    rt_change = post_rt - pre_rt
                    section.append(f"- Pre-distraction RT: {pre_rt}ms")
                    section.append(f"- Post-distraction RT: {post_rt}ms")
                    section.append(f"- RT change: {rt_change:+.0f}ms")
        
        return "\n".join(section)
    
    def _format_video_section(self, video_data: Dict) -> str:
        """Format video analysis data with full context."""
        
        section = ["\n### 3. REAL-TIME VIDEO ANALYSIS\n"]
        
        section.append("""**Analysis Context**:
During this stage, the participant watched an emotionally evocative video (a dramatic scene from the show "Hack") while their webcam captured facial responses. The recording was processed through 4 advanced ML models in real-time:

1. **Emotion Recognition Model** (Vision Transformer - ViT)
   - Analyzes facial expressions to detect 7 emotions: happy, sad, angry, fear, surprise, disgust, neutral
   - Processed at 3 FPS for accurate emotion classification
   - Includes XAI (Explainable AI) features: attention maps showing which facial regions influenced predictions

2. **Blink Detection Model** (MediaPipe Face Mesh)
   - Tracks eye aspect ratio (EAR) to detect blinks
   - Processed at 15 FPS for precise temporal detection
   - Normal blink rate: 15-20 per minute; elevated rates may indicate stress/anxiety

3. **Pupil Dilation Tracking** (MediaPipe Iris Landmarks)
   - Measures pupil size changes as indicators of autonomic nervous system arousal
   - Processed at 15 FPS for continuous monitoring
   - Dilation indicates cognitive load, emotional arousal, or stress response

4. **Gaze Estimation** (L2CS-Net Deep Learning Model)
   - Tracks where the person is looking (left, center, right)
   - Processed at 10 FPS for gaze direction classification
   - Attention to center indicates engagement; avoidance may indicate discomfort

**Trigger Video Description**:
The video shown was a dramatic, emotionally intense scene designed to elicit genuine emotional responses. It contains elements that may trigger sadness, fear, or distress, allowing assessment of emotional reactivity and regulation.

""")
        
        # Get summary with safe fallbacks
        summary = video_data.get('server_summary', video_data.get('summary', {}))
        timeline = video_data.get('timeline', [])
        duration = video_data.get('duration_seconds', len(timeline) / 30 if timeline else 0)
        
        section.append(f"\n**Recording Details**:")
        section.append(f"- Duration: {duration:.1f} seconds")
        section.append(f"- Total frames analyzed: {len(timeline)}\n")
        
        # Emotion Analysis
        emotion_summary = summary.get('emotion', {})
        if emotion_summary:
            section.append("\n**🎭 EMOTION ANALYSIS**:\n")
            
            dominant_emotion = emotion_summary.get('dominant_emotion', 'unknown')
            detection_rate = emotion_summary.get('detection_rate', 0)
            distribution = emotion_summary.get('distribution', {})
            
            section.append(f"- **Dominant Emotion**: {dominant_emotion.capitalize()}")
            section.append(f"- **Face Detection Rate**: {detection_rate:.1f}%\n")
            
            if distribution:
                section.append("**Emotion Distribution** (frame counts):")
                for emotion, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        percentage = (count / len(timeline)) * 100 if timeline else 0
                        section.append(f"  - {emotion.capitalize()}: {count} frames ({percentage:.1f}%)")
            section.append("")
        
        # Blink Analysis
        blink_summary = summary.get('blink', {})
        if blink_summary:
            section.append("\n**👁️ BLINK ANALYSIS**:\n")
            
            total_blinks = blink_summary.get('total_blinks', 0)
            blink_rate = blink_summary.get('blink_rate_per_minute', 0)
            avg_ear = blink_summary.get('avg_ear', 0)
            
            section.append(f"- **Total Blinks**: {total_blinks}")
            section.append(f"- **Blink Rate**: {blink_rate:.1f} blinks/minute")
            section.append(f"- **Average Eye Aspect Ratio (EAR)**: {avg_ear:.4f}")
            section.append(f"- **Clinical Interpretation**: Normal blink rate is 15-20/min. ")
            
            if blink_rate > 25:
                section.append(f"  Elevated rate ({blink_rate:.1f}/min) may indicate stress, anxiety, or eye strain.")
            elif blink_rate < 10:
                section.append(f"  Reduced rate ({blink_rate:.1f}/min) may indicate intense focus or attention.")
            else:
                section.append(f"  Rate is within normal range.")
            section.append("")
        
        # Pupil Analysis
        pupil_summary = summary.get('pupil', {})
        if pupil_summary:
            section.append("\n**👁️ PUPIL DILATION ANALYSIS**:\n")
            
            avg_size = pupil_summary.get('avg_pupil_size', 0)
            dilation_events = pupil_summary.get('pupil_dilation_events', 0)
            constriction_events = pupil_summary.get('pupil_constriction_events', 0)
            pupil_delta = pupil_summary.get('pupil_dilation_delta', 0)
            
            section.append(f"- **Average Pupil Size**: {avg_size:.4f} pixels")
            section.append(f"- **Dilation Events** (>10% increase): {dilation_events}")
            section.append(f"- **Constriction Events** (>10% decrease): {constriction_events}")
            section.append(f"- **Overall Dilation Delta**: {pupil_delta:.4f}")
            section.append(f"- **Clinical Interpretation**: Pupil dilation reflects autonomic arousal.")
            
            if dilation_events > 5:
                section.append(f"  Multiple dilation events ({dilation_events}) suggest emotional or cognitive arousal during video.")
            section.append("")
        
        # Gaze Analysis
        gaze_summary = summary.get('gaze', {})
        if gaze_summary:
            section.append("\n**👀 GAZE BEHAVIOR ANALYSIS**:\n")
            
            dominant_direction = gaze_summary.get('dominant_direction', 'unknown')
            attention_score = gaze_summary.get('attention_score', 0)
            distribution = gaze_summary.get('distribution_percentage', {})
            
            section.append(f"- **Dominant Gaze Direction**: {dominant_direction}")
            section.append(f"- **Attention Score** (% time looking at center): {attention_score*100:.1f}%\n")
            
            if distribution:
                section.append("**Gaze Distribution**:")
                section.append(f"  - Looking Left: {distribution.get('left', 0):.1f}%")
                section.append(f"  - Looking Center: {distribution.get('center', 0):.1f}%")
                section.append(f"  - Looking Right: {distribution.get('right', 0):.1f}%")
            
            section.append(f"\n- **Clinical Interpretation**: ")
            if attention_score > 0.7:
                section.append(f"High attention score ({attention_score*100:.1f}%) indicates sustained engagement with the video.")
            elif attention_score < 0.5:
                section.append(f"Lower attention score ({attention_score*100:.1f}%) may indicate discomfort, avoidance, or distraction.")
            else:
                section.append(f"Moderate attention score ({attention_score*100:.1f}%) indicates typical viewing behavior.")
        
        return "\n".join(section)
    
    def _generate_fallback_report(self, error: Exception) -> str:
        """Generate fallback report if Gemini API fails."""
        return f"""# Assessment Report Generation Failed

## Error
An error occurred while generating the AI-powered assessment report:

```
{str(error)}
```

## What This Means
The Gemini AI service encountered an issue. This could be due to:
- API connectivity issues
- Rate limiting
- Invalid API key
- Service unavailability

## Next Steps
1. Check your internet connection
2. Verify your GEMINI_API_KEY in the .env file
3. Try again in a few moments
4. If the issue persists, review the assessment data manually

## Your Assessment Data
Your assessment has been completed and saved successfully. The data is available in the dashboard for manual review.

---

*This is an automated fallback message. Please contact support if this issue continues.*
"""


def generate_ai_report(
    phq8_data: Optional[Dict] = None,
    game_data: Optional[Dict] = None,
    video_data: Optional[Dict] = None
) -> str:
    """
    Convenience function to generate AI assessment report.
    
    Args:
        phq8_data: PHQ-8 questionnaire results
        game_data: Game performance data
        video_data: Video analysis results
        
    Returns:
        Markdown-formatted report
    """
    analyzer = GeminiAnalyzer()
    return analyzer.generate_assessment_report(phq8_data, game_data, video_data)
