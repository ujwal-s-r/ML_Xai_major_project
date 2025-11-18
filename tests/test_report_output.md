# Assessment Report Generation Failed

## Error
An error occurred while generating the AI-powered assessment report:

```
429 Quota exceeded for quota metric 'Generate Content API requests per minute' and limit 'GenerateContent request limit per minute for a region' of service 'generativelanguage.googleapis.com' for consumer 'project_number:356579943112'. [reason: "RATE_LIMIT_EXCEEDED"
domain: "googleapis.com"
metadata {
  key: "service"
  value: "generativelanguage.googleapis.com"
}
metadata {
  key: "quota_unit"
  value: "1/min/{project}/{region}"
}
metadata {
  key: "quota_metric"
  value: "generativelanguage.googleapis.com/generate_content_requests"
}
metadata {
  key: "quota_location"
  value: "asia-southeast1"
}
metadata {
  key: "quota_limit"
  value: "GenerateContentRequestsPerMinutePerProjectPerRegion"
}
metadata {
  key: "quota_limit_value"
  value: "0"
}
metadata {
  key: "consumer"
  value: "projects/356579943112"
}
, links {
  description: "Request a higher quota limit."
  url: "https://cloud.google.com/docs/quotas/help/request_increase"
}
]
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
