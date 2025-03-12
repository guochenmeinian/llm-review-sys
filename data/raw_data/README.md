---
license: apache-2.0
---

# OpenReview Conference Papers with Reviews

## Dataset Overview

This dataset contains research papers and reviews crawled from OpenReview, covering top conferences such as ICML, NeurIPS, ICLR, and CVPR. The dataset includes:

- Paper metadata (title, conference, year, PDF URL)
- Review content (official review, meta review, official comment)
- Rating information (rating score, confidence score)

## Dataset Statistics (as of 2025-03-12)

| Metric | Value |
|--------|-------|
| Total Reviews | 120,818 |
| Time Span | 2023 - 2024 |
| Last Updated | 2025-03-12 00:52:07 |

## Data Structure

### Raw Data Format

Each conference directory contains a `results.json` file with the following structure:

```json
[
  {
    "venue_id": "conference.group/subgroup",
    "papers": [
      {
        "title": "Paper Title",
        "conference": "Conference Name",
        "pdf_url": "PDF URL",
        "reviews": [
          {
            "type": "Review Type",
            "content": {
              "comment": "Review Content"
            },
            "ratings": {
              "rating": "Rating Score",
              "confidence": "Confidence Score"
            }
          }
        ]
      }
    ]
  }
]
```

### Directory Structure
- `ICML/`, `NeurIPS/`, etc.: Raw data for each conference
