import json
import os
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration  
# ---------------------------------------------------------------------------
MODEL = "gemini-2.5-pro"
PROVIDER = "google"
TEMPERATURE = 0
TOP_P = 0.99
MAX_OUTPUT_TOKENS = 10000
REPAIR_LIMIT = 5
TIMEOUT_SEC = 3600

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent                        # llm_netlink_arena/api_gemini/
ARENA_DIR    = SCRIPT_DIR.parent / "NetLinkArena"           # competition repo root
PYTHON_EXEC  = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
PROMPT_PATH  = SCRIPT_DIR / "prompt_filled.md"
RUN_DIR      = SCRIPT_DIR / "run_gemini"

# Expected output from solution.py 
SUBMISSION_FILE = "predictions.csv"
EXPECTED_ROWS   = 1822   # NetLinkArena: 911 positive + 911 negative test edges


def extract_block(text: str, tag: str) -> str:
    """
    Extract content from <tag>...</tag>.
    Falls back to fenced code blocks if XML tags are missing.
    Follows the same pattern as llama_api_mode.py (CGCC).
    """
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if tag == "code":
            fenced = re.search(r"^```(?:python)?\s*(.*?)```$", content, re.DOTALL)
            if fenced:
                return fenced.group(1).strip()
        return content

    if tag == "code":
        fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            return fence.group(1).strip()

    if tag == "plan":
        plan_match = re.search(
            r"(?:^|\n)(?:#+\s*)?(?:Plan|Approach)\s*:?\s*(.*?)(?=```|<code>|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if plan_match and plan_match.group(1).strip():
            return plan_match.group(1).strip()

    raise ValueError(f"Missing <{tag}> block in model response")


def write_environment(run_dir: Path):
    """Save platform + pip freeze for reproducibility."""
    lines = [
        f"timestamp: {datetime.now().isoformat()}",
        f"platform: {platform.platform()}",
        f"python: {platform.python_version()}",
        f"model: {MODEL}",
        f"provider: {PROVIDER}",
        f"working_directory: {ARENA_DIR}",
    ]
    pip = subprocess.run(
        ["python", "-m", "pip", "freeze"],
        capture_output=True, text=True, check=False,
    )
    lines += ["\n[pip freeze]", pip.stdout]
    (run_dir / "environment.txt").write_text("\n".join(lines))


def validate_submission(arena_dir: Path) -> str:
    """
    Check that predictions.csv was written correctly.
    NetLinkArena requires: at minimum a CSV with predictions column.
    """
    csv_path = arena_dir / SUBMISSION_FILE
    if not csv_path.exists():
        return f"FAIL: {SUBMISSION_FILE} not found"

    lines = csv_path.read_text().strip().splitlines()
    if not lines:
        return f"FAIL: {SUBMISSION_FILE} is empty"

    header = lines[0].strip()
    if header != "id,y_pred":
        return f"FAIL: wrong header '{header}', expected 'id,y_pred'"

    n_rows = len(lines) - 1
    if EXPECTED_ROWS is not None and n_rows != EXPECTED_ROWS:
        return f"FAIL: expected {EXPECTED_ROWS} rows, got {n_rows}"

    # Check predictions are not all identical (collapsed model)
    preds = [line.split(",")[1] for line in lines[1:]]
    if len(set(preds)) == 1:
        return f"FAIL: all y_pred values are identical ({preds[0]}) — model collapsed, predictions must vary"

    return f"OK: {SUBMISSION_FILE} has {n_rows} rows with header '{lines[0]}'"


def call_llm(client: genai.Client, messages: list) -> str:
    """
    Single API call using the new google.genai SDK (replaces deprecated google.generativeai).
    New SDK: client.models.generate_content(model=..., contents=..., config=...)
    """
    config = types.GenerateContentConfig(
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        candidate_count=1,
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=messages,
        config=config,
    )
    try:
        return response.text or ""
    except Exception:
        return ""


def build_messages(history: list, new_user_text: str) -> list:
    """
    Build Gemini contents list from conversation history.
    New SDK uses types.Content with types.Part objects.
    Role must be 'user' or 'model' (not 'assistant').
    """
    messages = []
    for turn in history:
        role = "model" if turn["role"] == "assistant" else turn["role"]
        messages.append(
            types.Content(
                role=role,
                parts=[types.Part(text=turn["content"])],
            )
        )
    messages.append(
        types.Content(
            role="user",
            parts=[types.Part(text=new_user_text)],
        )
    )
    return messages


def main():
    # --- Check API key ---
    if "GOOGLE_API_KEY" not in os.environ:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set.\n"
            "Get a free key at https://aistudio.google.com/app/apikey and export:\n"
            "  export GOOGLE_API_KEY=your_key_here"
        )

    if not ARENA_DIR.exists():
        raise FileNotFoundError(
            f"NetLinkArena competition folder not found at: {ARENA_DIR}\n"
            f"Clone the repo there or update ARENA_DIR in this script."
        )

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"prompt_filled.md not found at: {PROMPT_PATH}\n"
            f"Copy the filled prompt template to: {PROMPT_PATH}"
        )

    RUN_DIR.mkdir(exist_ok=True)

    # --- Save config ---
    config = {
        "mode": "api_only",
        "provider": PROVIDER,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "repair_limit": REPAIR_LIMIT,
        "timeout_sec": TIMEOUT_SEC,
        "prompt_path": str(PROMPT_PATH),
        "arena_dir": str(ARENA_DIR),
        "submission_file": SUBMISSION_FILE,
    }
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))
    write_environment(RUN_DIR)

    # Copy prompt snapshot into run folder
    prompt_text = PROMPT_PATH.read_text()
    (RUN_DIR / "prompt_filled.md").write_text(prompt_text)

    # --- Configure Gemini client (new google.genai SDK) ---
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    # Conversation history (role/content pairs for easy management)
    history = []

    success = False
    repairs_used = 0
    total_start = time.time()

    for attempt in range(REPAIR_LIMIT + 1):
        print(f"\n=== Gemini attempt {attempt} | model={MODEL} ===")

        # Build Gemini-format messages from history + new user turn
        if attempt == 0:
            user_text = prompt_text
        # (repair text is already appended to history before next loop iteration)

        messages = build_messages(history[:-1] if attempt > 0 else [],
                                  history[-1]["content"] if attempt > 0 else prompt_text)

        # --- Call LLM ---
        text = call_llm(client, messages)
        (RUN_DIR / f"attempt_{attempt}_raw_response.txt").write_text(text)

        # Update history with assistant response
        if attempt == 0:
            history.append({"role": "user", "content": prompt_text})
        history.append({"role": "model", "content": text})

        # --- Parse plan and code ---
        try:
            plan = extract_block(text, "plan")
            code = extract_block(text, "code")
        except Exception as e:
            parse_error = str(e)
            print(f"Parse error: {parse_error}")
            (RUN_DIR / f"attempt_{attempt}_parse_error.txt").write_text(parse_error)
            history.append({
                "role": "user",
                "content": (
                    "Your response could not be parsed.\n\n"
                    f"Error: {parse_error}\n\n"
                    "Please return exactly:\n"
                    "<plan>\n...\n</plan>\n\n"
                    "<code>\n...\n</code>"
                ),
            })
            repairs_used = attempt + 1
            continue

        # Save per-attempt artifacts
        (RUN_DIR / f"attempt_{attempt}_plan.md").write_text(plan)
        (RUN_DIR / f"attempt_{attempt}_solution.py").write_text(code)

        # Write solution.py into competition repo root
        solution_path = ARENA_DIR / "solution.py"
        solution_path.write_text(code)

        # --- Execute solution.py ---
        print(f"Running solution.py from {ARENA_DIR} ...")
        start = time.time()
        result = subprocess.run(
            [PYTHON_EXEC, "solution.py"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
            check=False,
            cwd=str(ARENA_DIR),
        )
        elapsed = time.time() - start

        execution_log = (
            f"attempt={attempt}\n"
            f"returncode={result.returncode}\n"
            f"elapsed_sec={elapsed:.2f}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}\n"
        )
        (RUN_DIR / f"attempt_{attempt}_execution_log.txt").write_text(execution_log)
        (RUN_DIR / "execution_log.txt").write_text(execution_log)
        print(execution_log[:600])

        # --- Validate submission ---
        validation_log = validate_submission(ARENA_DIR)
        (RUN_DIR / f"attempt_{attempt}_validation_log.txt").write_text(validation_log)
        (RUN_DIR / "validation_log.txt").write_text(validation_log)
        print(f"Validation: {validation_log}")

        # --- Success? ---
        if result.returncode == 0 and validation_log.startswith("OK"):
            print("Success: predictions written correctly.")
            shutil.copyfile(ARENA_DIR / SUBMISSION_FILE, RUN_DIR / SUBMISSION_FILE)
            shutil.copyfile(solution_path, RUN_DIR / "final_solution.py")
            success = True
            repairs_used = attempt
            break

        # --- Repair: send traceback back ---
        if attempt < REPAIR_LIMIT:
            history.append({
                "role": "user",
                "content": (
                    "The script you produced failed.\n\n"
                    f"Return code: {result.returncode}\n\n"
                    f"Stdout:\n{result.stdout}\n\n"
                    f"Stderr / traceback:\n{result.stderr}\n\n"
                    f"Validation: {validation_log}\n\n"
                    "Please return a revised <plan> and <code>. "
                    "Address the error directly; do not restart from scratch unless necessary."
                ),
            })
        repairs_used = attempt + 1

    total_elapsed = time.time() - total_start

    # --- Summary CSV ---
    summary = (
        "mode,provider,model,success,repairs_used,temperature,top_p,"
        "max_output_tokens,total_elapsed_sec\n"
        f"api_only,{PROVIDER},{MODEL},{int(success)},{repairs_used},"
        f"{TEMPERATURE},{TOP_P},{MAX_OUTPUT_TOKENS},{total_elapsed:.2f}\n"
    )
    (RUN_DIR / "run_summary.csv").write_text(summary)
    print(f"\nTotal elapsed: {total_elapsed:.1f}s | success={success} | repairs={repairs_used}")

    if not success:
        raise RuntimeError(
            f"Run failed after {REPAIR_LIMIT} repair attempts. "
            f"Check {RUN_DIR} for logs."
        )


if __name__ == "__main__":
    main()
