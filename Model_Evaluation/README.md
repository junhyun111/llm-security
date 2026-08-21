# Model Evaluation

This directory is an isolated evaluation harness for the parent
`llm-security` project. It reads the parent implementation and the raw Juliet
dataset, but every generated index, frozen split, candidate file, artifact, and
report is written below this directory.

The initial workflow deliberately makes no LLM API calls:

1. index the sharded Juliet SARD packages;
2. freeze template/hash-disjoint train/dev/test assignments;
3. build a sanitized 100-scenario pilot (20 per active Expert);
4. run the production Semantic Analyzer and report Candidate Recall.

## Run

From `D:\llm-security` in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\Model_Evaluation\run.ps1 run-initial
```

The individual stages are also available:

```powershell
powershell -ExecutionPolicy Bypass -File .\Model_Evaluation\run.ps1 index
powershell -ExecutionPolicy Bypass -File .\Model_Evaluation\run.ps1 build-pilot
powershell -ExecutionPolicy Bypass -File .\Model_Evaluation\run.ps1 evaluate-analyzer
```

Use `--rebuild` on `index` or `run-initial` only when intentionally replacing
the generated evaluation index. The path guard refuses output locations outside
`Model_Evaluation`.

## Outputs

```text
work/index/juliet.sqlite       searchable raw-dataset index
work/index/index_summary.json dataset and exclusion statistics
work/frozen/split_manifest.json
work/pilot/cases_*.jsonl      sanitized cases consumed by the analyzer
work/pilot/pilot_review.csv   human-review sheet with raw and virtual locations
work/pilot/pilot_audit.json   leakage, label, and split checks
work/candidates/*.jsonl       frozen Semantic Analyzer candidates
results/analyzer_metrics.json Candidate Recall and safe-region candidate rate
```

Raw paths and Juliet labels are retained only in review/ground-truth metadata.
The inference source view removes comments and neutralizes filenames,
identifiers, macros, and strings that expose CWE, `bad`, `good`, `FLAW`, or
`FIX` labels while preserving line numbers.

The first completed run is summarized in [INITIAL_RESULTS.md](INITIAL_RESULTS.md).

## Train and evaluate with LLM APIs

Two executable notebooks now cover the paid experiment stages while keeping all
generated files inside `Model_Evaluation`:

- `train.ipynb`: materialize frozen train/dev splits, cache candidates, dry-run
  the Expert x model request matrix, collect resumable outcomes, audit matrix
  completeness, train the Utility Router, fit the escalation gate, and calibrate
  the recall-constrained threshold.
- `evaluation.ipynb`: materialize the untouched test split, collect its matching
  outcome matrix, compare Single/Fixed-2/Utility Top-2/Adaptive/Full-5, replay
  stage-wise end-to-end metrics, and optionally run live detection and patch
  verification.

Open JupyterLab from `D:\llm-security`:

```powershell
powershell -ExecutionPolicy Bypass -File .\Model_Evaluation\notebook.ps1 train
powershell -ExecutionPolicy Bypass -File .\Model_Evaluation\notebook.ps1 evaluation
```

The required Python/Jupyter packages are installed into the isolated
`Model_Evaluation/cache` directory by the launcher. The parent project's source
code and the raw dataset remain read-only inputs.

### Paid-call safety and resumption

API execution needs both locks:

1. set `RUN_PAID_EXPERIMENTS=1` in the parent `.env`;
2. set `EXECUTE_PAID=True` in the notebook only after inspecting the dry-run.

The notebook parameters `MAX_REQUESTS_PER_RUN`, `MAX_USD_PER_RUN`, and
`RESERVE_USD_PER_REQUEST` enforce a per-run guard. Every completed outcome is
appended immediately to JSONL. Re-running the cell resumes missing
`case/candidate/assignment` jobs instead of paying for completed jobs again.
Router training and final test evaluation refuse partial or duplicate matrices,
including candidates for which every assignment row is still missing.

`TRAIN_CASE_LIMIT`, `DEV_CASE_LIMIT`, and `TEST_CASE_LIMIT` default to `0`, which
means the complete frozen split. Set a positive value only for a cheaper smoke or
ablation run; use a separately named run directory/artifact when reporting such
results.

### What is measured

The saved reports include:

- Analyzer Candidate Recall and candidate-gate retention;
- validated finding Precision, truth Recall, F1, and exact case coverage;
- escalation recall, missed/unnecessary escalation rates, and Full-5 rate;
- average logical Expert count, request accounting, tokens, cost, and latency;
- Full-5 oracle retention versus the adaptive Router;
- optional patch apply rate and compile/test-verified repair rate.

Patch evaluation is disabled by default. Before enabling it, replace
`PATCH_VERIFICATION_COMMANDS` with commands that genuinely build and test a
Juliet package in the current environment. A diff that merely applies is not
counted as a verified repair. Patch application occurs only in a temporary copy;
`D:\llm-data` is never modified.

## New experiment outputs

```text
work/router_training/cases/       full or limited train/dev cases
work/router_training/candidates/  frozen semantic candidates
work/router_training/outcomes/    resumable Expert x model matrices
work/router_training/ledgers/     request/token/cost ledgers (no API key)
artifacts/juliet_utility_router.pkl
results/router_training/training_report.json
work/router_evaluation/           frozen test matrix and optional live results
results/router_evaluation/policy_and_end_to_end.json
```
