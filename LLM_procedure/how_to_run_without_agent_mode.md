# How to Run a GNN Competition with an LLM (No Agent Mode Required)

This guide explains how to reproduce the GNN-CB LLM evaluation using any LLM
(GPT, Llama, Mistral, etc.) without needing agent mode or any special
tools. You only need Python and an API key.

---

## Overview

The process has 3 steps:

```
Step 1: Clone the competition repo
Step 2: Fill the prompt file by hand
Step 3: Run GPT_script.ipynb
```

The script sends your prompt to the LLM, receives a solution, runs it, and
saves the predictions. If the code crashes, it automatically sends the error
back to the LLM and tries again (up to 5 times).

---

## Step 1 — Clone the Competition Repo

```bash
git clone https://github.com/<competition-repo>
cd <competition-repo>
```

You now have all the data files on your machine.

---

## Step 2 — Fill the Prompt File

Open `llm_real_or_fake/prompt.md`. It is the frozen prompt template used for
all 18 GNN-CB competitions. It has 5 blank slots.

Fill each slot by copying from the competition repo:

| Slot | What to paste |
|---|---|
| `{{README_VERBATIM}}` | Full contents of `README.md` |
| `{{TREE_OUTPUT}}` | Output of `find . -type f -not -path "./.git/*"` |
| `{{HEAD_AND_SHAPES}}` | Shapes and first few rows of each file in `data/` |
| `{{SUBMISSION_SPEC}}` | The "Submission Format" section of the README |
| `{{REQUIREMENTS_TXT}}` | Full contents of `requirements.txt` |

Save the filled file as `prompt_filled.md`.

This is a one-time manual step. No script needed — just copy and paste.

---

## Step 3 — Run the Script

Open `LLM/GPT_script.ipynb` and make 3 small changes:

1. Paste your API key
2. Set the path to `prompt_filled.md`
3. Run all cells

The script will do the following automatically:

```
1. Send prompt_filled.md to the LLM
2. LLM returns a <plan> and <code> block
3. Script extracts the code and saves it as solution.py
4. Script runs: python solution.py
5. Success → predictions.csv is written → done
   Error   → script sends the traceback back to the LLM
             LLM fixes the code → try again (max 5 attempts)
```

---

## Output

After the script finishes, results are saved to a folder called `run_gpt4o/`:

```
run_gpt4o/
├── plan.md             <- the plan produced by the LLM
├── solution.py         <- the code produced by the LLM
├── execution_log.txt   <- stdout and stderr of python solution.py
└── predictions.csv     <- the submission file for the leaderboard
```

---

## Works With Any LLM

| LLM | How to use |
|---|---|
| GPT-4o | `from openai import OpenAI` — use your OpenAI API key |
| Llama / Mistral via Ollama | Same OpenAI client, set `base_url="http://localhost:11434/v1"` |
| HuggingFace models | Use `transformers.pipeline("text-generation", ...)` |

Only the first 3 lines of the script change. The prompt, the repair loop, and
the output are identical for all LLMs.

