# Retrieval Test

Collection used: `buraq_source_memory_lab2`  
Embedding model: `all-MiniLM-L6-v2`  
Indexed chunks: `30`

## Test 1: Submission checklist lookup

Query:
`What files are required in the Lab 1 submission checklist?`

Metadata filter:
`None`

Top result:
- `capstone_lab_brief_extracted-section-2`
- Metadata: `doc_type=course_brief`, `department=academics`, `priority_level=high`
- Retrieved content:

```text
Document: capstone lab brief extracted
Section: submission checklist
- PRD.md with problem statement, user personas, and measurable success metrics
- Architecture_Diagram.png showing the end-to-end system design
- Initial_Data folder with 3 to 5 representative raw files for indexing in Lab 2
```

Why this is correct:
The query asks for required submission files, and the top hit is the exact checklist section from the course brief rather than a loosely related email or note.

## Test 2: Metadata filtering for recruitment-only retrieval

Query:
`Which email asks for an updated resume before an interview?`

Metadata filter:
`{"department": "careers"}`

Why filtering matters:
Without filtering, the first result was a deadline record derived from the same event. After filtering to the `careers` department, the search returned the original recruitment email itself.

Top result:
- `inbox_emails_sample-msg_002`
- Metadata: `doc_type=incoming_email`, `department=careers`, `priority_level=high`
- Retrieved content:

```text
Subject: Shortlisted for AI Intern interview
From: Talent Team
To: hamza@student.edu
Date: 2026-05-02 11:42
Snippet: Congratulations. Please confirm your interview availability and upload your updated resume before May 6.
Category: recruitment
Priority: high
Action required: yes
Has attachment: yes
```

Why this is correct:
This is the exact inbox message that contains the resume-upload instruction. The metadata filter improves precision by excluding style-reference emails and task-summary rows.

## Test 3: Actionable task retrieval from deadline records

Query:
`What task says to simplify the sponsor demo storyline?`

Metadata filter:
`{"doc_type": "deadline_record"}`

Top result:
- `project_deadlines-deadline_006`
- Metadata: `doc_type=deadline_record`, `department=task_management`, `priority_level=high`
- Retrieved content:

```text
Title: Simplify sponsor demo storyline
Due date: 2026-05-06 18:00
Owner: Hamza Ali
Source: meeting
Details: Reduce jargon in dashboard explanation and emphasize user impact.
Status: open
Linked email id: msg_007
```

Why this is correct:
The query asks for an actionable task, and the filter restricts retrieval to structured task records. That gives the agent the due date, owner, and status in one grounded chunk instead of forcing it to infer a task from free-form email text.
