# GradPath retrieval eval report

- Date: 2026-09-28
- Questions: 105 (BIOL 1306 · Biology: 15, BIOL 2321 · Microbiology: 15, CHEM 1311/1312 · Chemistry: 15, MATH 1342 · Statistics: 15, MATH 2413 · Calculus I: 15, PHYS 1301 · Physics: 15, PSYC 2301 · Psychology: 15)
- Chunks in index: 7856
- k = 5; relevance = chapter-number match (fallback: title substring)
- recall@5 denominator = corpus chunks in the expected chapter (per course), precomputed per course; 0.0 when the corpus has no chunk labeled with the expected chapter

| subject                         | n   | precision@5 | recall@5  | mrr       | keyword_coverage@5 |
|---------------------------------|-----|-------------|-----------|-----------|--------------------|
| BIOL 1306 · Biology             | 15  | 0.733       | 0.125     | 0.833     | 0.844              |
| BIOL 2321 · Microbiology        | 15  | 0.613       | 0.076     | 0.680     | 0.867              |
| CHEM 1311/1312 · Chemistry      | 15  | 0.893       | 0.105     | 0.967     | 0.889              |
| MATH 1342 · Statistics          | 15  | 0.840       | 0.083     | 1.000     | 0.867              |
| MATH 2413 · Calculus I          | 15  | 0.853       | 0.049     | 0.889     | 0.911              |
| PHYS 1301 · Physics             | 15  | 0.920       | 0.095     | 0.900     | 0.844              |
| PSYC 2301 · Psychology          | 15  | 0.747       | 0.075     | 0.867     | 0.867              |
| **macro avg** (subjects equal)  | 105 | **0.800**   | **0.087** | **0.877** | **0.870**          |
| **micro avg** (questions equal) | 105 | **0.800**   | **0.087** | **0.877** | **0.870**          |

## Notes

- Chapter labels come from PDF-outline extraction (per-book coverage 86.7-99.4%); remaining non-chapter labels are honest back-matter (Preface, Glossary, Index, appendices).
- recall@5's denominator is EVERY corpus chunk in the expected chapter (often hundreds), so it is structurally low by design -- treat precision@5 / MRR / keyword_coverage@5 as the headline metrics.
- keyword_coverage@5 is label-independent: fraction of distinctive expected-chapter phrases found in the top-5 chunks' text.
