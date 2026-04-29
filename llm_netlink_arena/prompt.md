# GNN-CB Frozen Prompt Template (v1.0)

This template is used **unchanged** for every one of the 18 competitions in the
GNN Coding Benchmark (GNN-CB). Only the slot contents (README, repo tree, data
sample, submission spec) differ across tasks because those *are* the task. No
per-competition prompt tuning is performed.

The structure follows the **plan-then-code** pattern of Jiang et al. (2024,
"Self-Planning Code Generation"), the **execute-and-repair** loop of Yang et al.
(2024, "SWE-agent"), and the **single-template / slot-filled** evaluation
protocol of Zhuo et al. (2024, "BigCodeBench") and Jain et al. (2024,
"LiveCodeBench").

The template was designed and frozen using a single pilot competition (C03
Real_Or_Fake) prior to evaluation on the remaining 17 competitions.

---

## SYSTEM PROMPT (identical for all 18 competitions)

```
Given a GNN
coding competition consisting of:
  (1) the competition README,
  (2) the repository file tree,
  (3) a data-sample summary (file shapes, dtypes, head() of CSVs),
  (4) the required submission format.

Your task is to produce a single, self-contained Python script that, when
executed from the repository root in a CPU-only environment, trains a GNN,
generates predictions for the test set, and writes the submission file at the
location specified by the README.

You must respond in exactly this format:

<plan>
A 5-10 bullet plan covering: which features to use, model architecture,
training protocol, validation strategy, threshold/decoding, and how the
submission file will be written.
</plan>

<code>
A complete, runnable Python script. Constraints:
  - Use only the libraries declared in requirements.txt of the repo
    (plus the Python standard library).
  - The script must be runnable as `python solution.py` from the repo root.
  - Set seeds (numpy, torch, random) for reproducibility.
  - The model must be a Graph Neural Network (per competition rules).
  - The script must complete on CPU within 60 minutes for this competition.
  - The script must write the submission file in exactly the format the
    competition README specifies.
  - Do not download external data; use only files already on disk.
</code>

After the script is executed, you may receive an error trace. If so, return a
revised <plan> and <code> that addresses the specific error. You have at most
5 repair iterations.
```

## USER PROMPT SCAFFOLD (same template, slots filled per competition)

```
## Competition README
{{README_VERBATIM}}

## Repository tree
{{TREE_OUTPUT}}

## Data sample
{{HEAD_AND_SHAPES}}

## Required submission format
{{SUBMISSION_SPEC_FROM_README}}

## Allowed libraries
{{REQUIREMENTS_TXT}}

Produce <plan> then <code>.
```

---

## How slots are filled (mechanical, no human judgment)

| Slot | Fill rule |
|---|---|
| `{{README_VERBATIM}}` | Verbatim contents of `README.md` at the repo root |
| `{{TREE_OUTPUT}}` | `find . -type f -not -path './.git/*'` truncated to 200 lines |
| `{{HEAD_AND_SHAPES}}` | For each CSV/NPY/NPZ in `data/`: shape, dtype, and `.head(5)` if tabular |
| `{{SUBMISSION_SPEC_FROM_README}}` | Section of README under "Submission" / "Output" headers, verbatim |
| `{{REQUIREMENTS_TXT}}` | Verbatim contents of `requirements.txt` |

No human curation. No hints. No per-task examples.

---

## Repair loop

If the generated script errors during execution, the following message is sent
back (also fixed across all competitions):

```
The script you produced failed with this traceback:

<traceback>
{{TRACEBACK}}
</traceback>

The current state of any files you wrote:

<state>
{{LIST_OF_FILES_CREATED}}
</state>

Please return a revised <plan> and <code>. Address the error directly; do not
restart from scratch unless necessary.
```

Maximum 5 repair iterations. If still failing after 5, the run is recorded as
a failure with the final traceback.

---

## What is reported

For each (model, competition) pair we report:
- Whether the script ran to completion
- Number of repair iterations used
- Wall-clock time (CPU)
- Validation-set metrics (computed locally with the repo's own evaluation code)
- Test-set predictions file (for optional submission to leaderboard)

This template is the *only* prompt artifact released. Reproducing GNN-CB LLM
results requires only this template plus the repo URLs.
