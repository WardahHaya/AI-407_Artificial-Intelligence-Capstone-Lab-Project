# Lab 7 Evaluation Report

Metric framework: local DeepEval-style deterministic audit using grounded reference answers, semantic similarity, and exact tool-call matching.

Environment note: no external judge-model or LangSmith API key was configured locally, so the evaluation pipeline used local semantic scoring with the same embedding stack that powers grounding.

## Average Scores

| Metric | Average Score |
| --- | ---: |
| Average Faithfulness | 0.952 |
| Average Answer Relevancy | 0.690 |
| Average Tool Call Accuracy | 1.000 |

## Lowest-Scoring Cases

| Case ID | Expected Tool | Actual Tool | Faithfulness | Relevancy | Tool Accuracy |
| --- | --- | --- | ---: | ---: | ---: |
| case_18_slides_deadline | search_knowledge_base | search_knowledge_base | 0.862 | 0.522 | 1.000 |
| case_14_sponsor_feedback | search_knowledge_base | search_knowledge_base | 0.950 | 0.454 | 1.000 |
| case_19_style_reference | search_knowledge_base | search_knowledge_base | 0.882 | 0.567 | 1.000 |

## Findings

- Retrieval-grounded questions scored highest on faithfulness because their answers stayed close to the indexed context.
- Draft-generation cases had slightly lower relevancy because the tool returns a full formatted email while the gold answer is a concise reference summary.
- Tool-call accuracy remained high because the evaluator matched explicit routing rules against the expected lab tool inventory.
