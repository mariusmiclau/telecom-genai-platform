# Golden retrieval evaluation

Dataset: `tickets-v2.0-ticket-aware-chunking`  
Golden questions: 50  
Embedding: `telecom-hashing-embedding-v1` (384 dimensions)

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 | Evidence coverage@5 | p95 latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_recursive_vector | 64.0% | 78.0% | 80.0% | 0.690 | 0.710 | 71.4% | 6.134 |
| B_ticket_aware_hybrid | 74.0% | 86.0% | 96.0% | 0.796 | 0.802 | 80.7% | 8.979 |
| C_ticket_aware_hybrid_reranker | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 | 90.0% | 18.389 |

## Recall@5 by question tag

| Configuration | acronym | entity_id | semantic |
|---|---:|---:|---:|
| A_recursive_vector | 70.0% | 53.3% | 96.0% |
| B_ticket_aware_hybrid | 80.0% | 66.7% | 100.0% |
| C_ticket_aware_hybrid_reranker | 100.0% | 100.0% | 100.0% |

> These are measured offline results for the committed synthetic dataset. They are not production claims. Re-run the command below after retrieval or dataset changes.

```bash
python -m src.evaluation.cli --compare-to reports/evaluation/baseline.json
```
