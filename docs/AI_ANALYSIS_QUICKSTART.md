# 🤖 AI Analysis Feature - Quick Start Guide

## Prerequisites

1. **GEMINI_API_KEY** must be set in `.env` file (root directory)
2. **python-dotenv** must be installed
3. Complete assessment data from all 3 stages (recommended)

## Installation

```powershell
# Navigate to project root
cd C:\Users\ujwal\OneDrive\Documents\Github\ML_Xai_major_project

# Install required package
uv pip install python-dotenv

# Or install all dependencies
uv pip install -r requirements.txt
```

## Starting the Server

```powershell
# Navigate to backend
cd backend

# Start with uvicorn
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Usage Flow

### 1. Complete Assessment
Open browser: **http://127.0.0.1:8000**

1. **PHQ-8 Questionnaire** - Answer 8 depression screening questions
2. **Buzzer Bombardment Game** - Play 60-second cognitive test
3. **Video Analysis** - Watch trigger video while being recorded

### 2. Access AI Analysis

After video analysis completes, you'll see two buttons:
- 📊 **View Dashboard** - See visualizations
- 🤖 **AI Analysis** - Generate comprehensive report

Click **AI Analysis** button.

### 3. Wait for Report Generation

- Loading screen appears with spinner
- Gemini AI analyzes your complete assessment
- Takes 10-30 seconds
- Progress is shown

### 4. Read Your Report

The AI-generated report includes:

✅ **Executive Summary**  
✅ **PHQ-8 Depression Assessment** (detailed interpretation)  
✅ **Cognitive Performance Analysis** (game metrics)  
✅ **Behavioral & Emotional Analysis** (video insights)  
✅ **Integrated Clinical Interpretation**  
✅ **Personalized Recommendations**  
✅ **Important Disclaimers**

### 5. Download or Print

- **📥 Download Report** - Saves as Markdown (.md) file
- **🖨️ Print Report** - Print-optimized layout
- **📊 View Dashboard** - Return to dashboard

## API Endpoint

### Generate AI Report

```
POST /api/ai-analysis/generate
Content-Type: multipart/form-data

Body:
  session_id: <your-session-id>

Response:
{
  "session_id": "uuid",
  "report": "# Mental Health Assessment Report\n\n...",
  "generated_at": "2025-11-17T10:30:00Z"
}
```

### Example using curl

```bash
curl -X POST http://127.0.0.1:8000/api/ai-analysis/generate \
  -F "session_id=your-session-id-here"
```

### Example using JavaScript

```javascript
const formData = new FormData();
formData.append('session_id', sessionId);

const response = await fetch('/api/ai-analysis/generate', {
    method: 'POST',
    body: formData
});

const data = await response.json();
console.log(data.report); // Markdown report
```

## Troubleshooting

### Error: "GEMINI_API_KEY not found"
**Solution**: Add your API key to `.env` file in project root:
```env
GEMINI_API_KEY=your-api-key-here
```

### Error: "No assessment data found"
**Solution**: Complete at least one assessment stage (preferably all 3)

### Error: "Module 'dotenv' not found"
**Solution**: Install python-dotenv:
```powershell
uv pip install python-dotenv
```

### Report takes too long
**Solution**: 
- Normal processing time: 10-30 seconds
- Check internet connection
- Verify Gemini API is accessible
- Check for rate limiting

### Import errors
**Solution**: Ensure all dependencies are installed:
```powershell
uv pip install -r requirements.txt
```

## Features

### ✨ Comprehensive Analysis
- Analyzes PHQ-8, game performance, and video data together
- Provides integrated insights across all domains
- Clinical interpretation with research-based norms

### 📝 Detailed Reporting
- 1500-2500 word comprehensive report
- Markdown formatted for easy reading
- Professional clinical language
- Compassionate and actionable

### 🎯 Personalized Recommendations
- Immediate action items
- Professional support suggestions
- Evidence-based self-care strategies
- Follow-up timeline

### 🔒 Privacy & Security
- All processing happens with your session data
- No data shared beyond Gemini API call
- Reports can be downloaded for offline use
- No data persistence beyond JSONL files

## Example Report Structure

```markdown
# Mental Health Assessment Analysis Report

## Executive Summary
Brief overview and key findings...

## 1. PHQ-8 Depression Assessment
- Total Score: 12/24 (Moderate)
- Question-by-question analysis
- Clinical significance

## 2. Cognitive Performance Analysis
- Accuracy: 85%
- Reaction Time: 1245ms
- Impulsivity indicators
- Distraction impact

## 3. Behavioral & Emotional Analysis
- Dominant emotion: Sad (45% of frames)
- Blink rate: 22/min (elevated)
- Pupil dilation events: 8
- Attention score: 72%

## 4. Integrated Clinical Interpretation
How the findings connect across domains...

## 5. Recommendations
Specific, actionable next steps...

## 6. Important Disclaimers
Screening tool notice, professional consultation...
```

## Tips for Best Results

1. **Complete all 3 stages** for comprehensive analysis
2. **Be honest** in PHQ-8 responses
3. **Focus during game** for accurate cognitive metrics
4. **Face camera directly** during video analysis
5. **Express naturally** during trigger video

## Support

For issues or questions:
1. Check console logs in browser (F12)
2. Check server logs in terminal
3. Review the full documentation in `/docs`
4. Verify API key is valid

## Technical Details

- **Model**: Gemini 2.0 Flash Experimental
- **Prompt Length**: ~3000-5000 characters (detailed)
- **Response Format**: Markdown
- **Processing Time**: 10-30 seconds
- **Cost**: 1 API call per report generation

---

**Status**: ✅ Ready to Use  
**Version**: 1.0  
**Last Updated**: November 17, 2025
