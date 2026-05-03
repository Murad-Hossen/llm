# How to Run a GNN Competition Evaluation Using Agent Mode

This guide explains how any AI coding agent was used to produce results like
`llm_real_or_fake/run_claude/run_summary.csv` — and how anyone can repeat
the same process for any competition using their preferred tool.

---

## What Is Agent Mode?

Agent mode means the AI can read files, run commands, and fix its own errors
on its own — without you doing anything manually. Many tools support this today:

| Tool | Agent Mode Name |
|---|---|
| Claude Code (CLI) | Agent mode (default) |
| Cursor | Composer / Agent |
| VS Code + Copilot | Copilot Edits / Agent |
| Windsurf | Cascade |
| Aider | Default mode |
| OpenHands | Full agent |

They all work the same way for this task.

---

## What Agent Mode Does

When you give the agent a competition repo, it does everything on its own —
no manual steps needed:

```
1. Reads README.md               ← understands the task
2. Reads the repo file tree       ← sees what files exist
3. Reads data files               ← checks shapes and formats
4. Reads requirements.txt         ← knows which libraries to use
5. Writes a plan                  ← decides architecture and training
6. Writes solution.py             ← complete Python script
7. Runs python solution.py        ← executes the code
8. If error → fixes and reruns    ← up to 5 repair attempts
9. Saves predictions.csv          ← the submission file
```

You only give it one instruction. It handles everything else.

---

## What You Need

- Any AI agent tool (Claude Code, Cursor, VS Code Copilot, Aider, etc.)
- An API key for the underlying LLM (Claude, GPT, Gemini, etc.)
- The competition repo cloned locally

---

## The One Instruction You Give

Open your agent tool, point it at the competition repo, and give it this:

```
Read the README, explore the data folder, then write and run a
solution.py that trains a GNN and saves predictions.csv.
Use only the libraries in requirements.txt.
Follow this frozen prompt template: /path/to/llm_real_or_fake/prompt.md
```

That is the entire instruction. The agent does the rest.

---

## Tool-Specific Instructions

### Claude Code (CLI)
```bash
cd /path/to/Real_Or_Fake
claude "Read the README, explore the data, write and run solution.py 
that trains a GNN. Follow the template at ../llm_real_or_fake/prompt.md"
```

### Cursor
1. Open the competition repo folder in Cursor
2. Press `Ctrl+Shift+I` to open Composer (Agent mode)
3. Type the instruction above and press Enter

### VS Code + GitHub Copilot
1. Open the repo folder in VS Code
2. Open Copilot Chat (`Ctrl+Shift+P` → "Copilot: Open Chat")
3. Switch to **Agent** mode in the chat panel
4. Type the instruction above

### Aider (terminal)
```bash
cd /path/to/Real_Or_Fake
aider --model gpt-4o \
  --message "Read the README, explore data/, write and run solution.py 
  that trains a GNN following ../llm_real_or_fake/prompt.md"
```

---

## What the Agent Produces

After it finishes, you will find these files:

```
run_<agentname>/
├── plan.md             ← the plan (architecture, training protocol)
├── solution.py         ← the complete Python script the agent wrote
├── execution_log.txt   ← stdout and stderr of python solution.py
├── predictions.csv     ← test-set predictions, ready to submit
└── run_summary.csv     ← val_f1, val_acc, threshold, epochs, splits
```

The `run_summary.csv` for Real_Or_Fake (Claude agent) looked like this:

```
val_f1,   val_acc,  threshold, epochs_run, n_train, n_val, n_test
0.9768,   0.9762,   0.53,      20,         3826,    546,   1092
```

---

## How This Compares to Running Without Agent Mode

| | Agent mode | Without agent mode |
|---|---|---|
| Read README | Agent does it automatically | You paste it into prompt_filled.md |
| Get repo tree | Agent does it automatically | You run `find .` and paste output |
| Sample data files | Agent does it automatically | You check shapes and paste them |
| Fill prompt slots | Agent does it automatically | You do it manually |
| Write solution.py | Agent does it | LLM does it via API script |
| Run solution.py | Agent does it | Your script does it via subprocess |
| Fix errors | Agent does it | Script sends traceback back to LLM |

Agent mode removes every manual step. Without agent mode, the only extra work
is filling the prompt file once before running the script.

The LLM receives exactly the same information either way.

---

## The Frozen Prompt Template

Both approaches use the same frozen template at `llm_real_or_fake/prompt.md`.
It is never edited between competitions. The only thing that changes is the
slot values — README, tree, data shapes, submission spec, requirements.

In agent mode, the agent fills those slots itself by reading the repo.
Without agent mode, you fill them manually into `prompt_filled.md`.

---

## Reproducibility

To reproduce any GNN-CB result with any agent tool:

1. Clone the competition repo
2. Give the agent the repo path and the frozen prompt template
3. Record: val_acc, val_f1, repairs used, wall-clock time

The frozen template is the only prompt artifact. No per-task tuning, no
hints, no examples. The same template was used for all 18 competitions.
