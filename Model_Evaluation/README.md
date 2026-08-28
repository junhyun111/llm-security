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

Two executable notebooks cover the API experiment stages while keeping all
generated files inside `Model_Evaluation`:

- `train.ipynb`: preserve the frozen splits, create deterministic stratified
  Train 6,000 / Dev 1,500 cohorts, preserve every E6 case, prioritize CWE and
  leakage-family diversity, train LR/GBDT/small-MLP Candidate Rankers by dev
  Recall@1/2/4/8, collect resumable batched outcomes, calibrate per-Expert
  validation thresholds on Dev, and train the Utility Router.
- `evaluation.ipynb`: evaluate all three Router variants on the complete frozen
  Test split of 8,337 scenarios, report micro/per-Expert/macro metrics, and run
  optional batched patch verification for ground-truth-matched findings.

Open JupyterLab from `D:\llm-security`:

```powershell
powershell -ExecutionPolicy Bypass -File .\Model_Evaluation\notebook.ps1 train
powershell -ExecutionPolicy Bypass -File .\Model_Evaluation\notebook.ps1 evaluation
```

The project does not impose a Python version or create a virtual environment
automatically. Create your own environment at `Model_Evaluation\.venv` or the
project root `.venv` (the launcher detects both) and install the packages from
`requirements.txt`. The
parent project's source code and the raw dataset remain read-only inputs.

In VS Code, select this interpreter for the notebooks:

```text
D:\llm-security\.venv\Scripts\python.exe
```


### API execution and resumption

Only `OPENROUTER_API_KEY` is required in the parent `.env`. There is no separate
paid-run flag in the notebooks. All candidates and the five logical Expert tasks
for one Juliet case are sent through `BatchedExpertRunner` in one physical API
request. The five Expert outcomes remain separately attributed for Utility
Router training, but they are not five separate API calls.

Each completed case is atomically checkpointed. Re-running the collection cell
skips completed cases only when the model, prompt version, candidate IDs,
feature schema, and validator thresholds match. A prompt/schema/policy change
therefore invalidates stale case checkpoints instead of silently mixing
experiments. Router training and final test evaluation refuse partial or
duplicate matrices.

Candidate extraction and Candidate Ranker training are local and make no LLM
API calls. API calls begin only in the batched Expert outcome cell. The selected
ranker is fitted on Train, selected on Dev Recall@4, and then applied unchanged
to Test.

Outcome collection uses an `asyncio.Semaphore` completion pool with
`MAX_CONCURRENCY = 1000`. It does not wait for a fixed chunk to finish: whenever
one request completes, the next waiting case starts immediately. Transient 429
and 5xx/network failures use exponential backoff with deterministic jitter, and
every successful case is checkpointed before the pool continues. A concurrency
of 1,000 is intentionally aggressive; lower `MAX_CONCURRENCY` in either
notebook if the OpenRouter account or selected provider has a smaller rate or
connection limit.

The cohort policy is frozen in `configs/cohort_15837.toml`:

```text
Train 6,000: E1 1,800 / E3 1,300 / E4 1,700 / E5 1,128 / E6 72 (all)
Dev   1,500: E1   450 / E3   325 / E4   425 / E5   282 / E6 18 (all)
Test  8,337: complete leakage-disjoint frozen test split
```

Sampling never consults analyzer candidates. It first covers every available
CWE, then unseen leakage groups, then additional family variations. Every
selection and its sampling weight is recorded in a deterministic manifest.

### What is measured

The saved reports include:

- Analyzer Candidate Recall and candidate-gate retention;
- Candidate Ranker Recall@1, Recall@2, Recall@4, and Recall@8;
- raw Expert Recall, validator true-finding retention, false-finding rejection,
  and final validated Recall as separate stage metrics;
- validated finding Precision, truth Recall, F1, and exact case coverage;
- escalation recall, missed/unnecessary escalation rates, and Full-5 rate;
- average logical Expert count, request accounting, tokens, cost, and latency;
- Full-5 oracle retention versus the adaptive Router;
- shared 128→64 multi-task MLP Router metrics;
- MLP Router uses CUDA automatically when the installed PyTorch build detects a GPU;
- Brier score and expected calibration error for Router probabilities;
- optional nested 1k/2k/4k/6k MLP learning curves with E6 preserved;
- optional patch apply rate and compile/test-verified repair rate.

The Expert prompt receives an evidence-local code slice, evidence/value-flow
graph, available type/conversion facts, and bounded direct caller/callee
summaries. Each Expert follows a domain-specific proof obligation. Missing
static CWE support or an Expert-specific confidence miss is recorded as
`uncertain`, while impossible locations and invalid evidence IDs remain hard
rejections.

### GPU MLP training

The training notebook runs only the multi-task MLP; LR and GBDT are not fitted.
It uses batch size 512, initial learning rate 0.002, at most 100 epochs, and
early-stopping patience 12. `ReduceLROnPlateau` halves the learning rate after
four stale validation epochs down to 0.00001. Every epoch prints the current
learning rate, train/validation loss, best epoch, stale count, and elapsed time.

The multi-task MLP selects CUDA automatically. For this Windows RTX 3060 environment
install the CUDA-enabled PyTorch build into the selected virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch==2.13.0 --index-url https://download.pytorch.org/whl/test/cu126
```

Restart the notebook kernel afterwards. The setup cell must print `MLP training
device: cuda`; `cpu (CUDA unavailable)` means that a CPU-only PyTorch build is
still active.

Patch evaluation is disabled by default. Before enabling it, replace
`PATCH_VERIFICATION_COMMANDS` with commands that genuinely build and test a
Juliet package in the current environment. A diff that merely applies is not
counted as a verified repair. Patch application occurs only in a temporary copy;
`D:\llm-data` is never modified.

## New experiment outputs

```text
work/cohort_15837/                 deterministic cohort manifests and audit
work/router_training_stratified_7500/cases/
work/router_training_stratified_7500/candidates/
work/router_training_stratified_7500/outcomes/
work/router_training_stratified_7500/ledgers/
artifacts/juliet_utility_router.pkl
artifacts/juliet_utility_router_multitask_mlp.pkl
artifacts/candidate_ranker/candidate_ranker.pkl
artifacts/candidate_ranker/candidate_ranker_{logistic_regression,gradient_boosting,small_mlp}.pkl
results/router_training_stratified_7500/training_report.json
results/router_training_stratified_7500/candidate_ranker_report.json
results/router_training_stratified_7500/validator_calibration.json
results/router_training_stratified_7500/learning_curves.json
work/router_evaluation_full_test/
results/router_evaluation_full_test/evaluation_*.json
```
