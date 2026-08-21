# Initial Juliet Evaluation Results

- Snapshot date: 2026-08-21
- Configuration: `configs/pilot.toml`
- Seed: `2026`
- LLM API calls: `0`

## Dataset preparation

- raw SARD packages scanned: 64,099
- supported five-Expert scenarios: 55,224
- leakage groups: 1,322
- split sizes: train 38,563 / dev 8,324 / test 8,337
- template/exact-hash/canonical-hash split audit: passed
- dataset fingerprint:
  `a4d91ea87f2e957bbbd3324fbaec19b6e74a91c413436ba84e73ffed7f0d1b08`

Supported scenario distribution:

| Expert | Scenarios |
|---|---:|
| E1 Memory Safety | 27,510 |
| E3 Integer / Size / Type | 11,016 |
| E4 Taint / API Contract | 14,274 |
| E5 Control / State / Error | 2,316 |
| E6 Concurrency / TOCTOU | 108 |

## Pilot audit

- 100 scenarios: 20 per Expert
- train/dev/test: 70/15/15
- SARIF positive regions: 102
- paired target-CWE-safe regions: 229
- oracle label leaks after sanitization: 0
- label and region audit: passed
- pilot split/hash leakage audit: passed
- materialization warnings: 0

## Semantic Analyzer

- analyzed cases: 100
- analysis failures: 0
- candidates: 343
- ground-truth hits: 65 / 102
- Candidate Recall: **63.73%**
- average candidates per case: 3.43
- mean/median candidate size: 24.59 / 16 LOC
- average analysis latency: approximately 5.2 ms per case

Candidate Recall by target Expert:

| Expert | Hits / GT | Recall |
|---|---:|---:|
| E1 Memory Safety | 20 / 22 | 90.91% |
| E3 Integer / Size / Type | 8 / 20 | 40.00% |
| E4 Taint / API Contract | 13 / 20 | 65.00% |
| E5 Control / State / Error | 6 / 20 | 30.00% |
| E6 Concurrency / TOCTOU | 18 / 20 | 90.00% |

The Analyzer created candidates in 148 of 229 target-CWE-safe function regions
(64.63%). This is a review-load/negative-candidate measure, not an LLM false
positive rate. Juliet good paths can contain locally suspicious operations that
are safe only because of an upstream good source, guard, or call path. A false
positive is measured later from validated Expert findings over the full safe
path.
