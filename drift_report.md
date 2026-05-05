# Drift Report

Negative feedback rows analyzed: 6

## Category Breakdown

| Category | Count | Share |
| --- | ---: | ---: |
| Wrong Tone | 2 | 33.3% |
| Missing Context | 2 | 33.3% |
| Tool Error | 1 | 16.7% |
| Other | 1 | 16.7% |

## Findings

- The largest cluster of negative feedback points to the agent giving vague or incomplete answers when the prompt needed more grounded detail.
- A smaller but meaningful set of failures comes from tone mismatch in drafted replies, especially when the user expected a warmer or more concise response.
- Tool-related issues are mostly tied to insufficiently explicit evidence handoff, which can make the final answer less specific than the user expects.

## Example Failed Rows

| message_id | Category | User Input | Comment |
| --- | --- | --- | --- |
| demo-msg-002 | Wrong Tone | Write a warm reply to Talent Team. | Too formal and not warm enough. |
| demo-msg-003 | Missing Context | What is the deadline linked to the recruiter interview email? | Too vague and missed the grounded deadline context. |
| demo-msg-005 | Tool Error | Find the email about the evaluation rubric. | Tool failed and the answer was incomplete. |
| demo-msg-006 | Missing Context | Summarize my emails from the last 2 days. | Missed important details and felt too generic. |
| demo-msg-008 | Wrong Tone | Draft a friendly email to Areeba about the architecture slide. | Wrong tone. It should sound friendly, not stiff. |
