import json
import os
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from together import Together

# ---------------------------------------------------------------------------
# Configuration  
# ---------------------------------------------------------------------------
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
PROVIDER = "together"
TEMPERATURE = 0
TOP_P = 0.99
MAX_OUTPUT_TOKENS = 4000
REPAIR_LIMIT = 5
TIMEOUT_SEC = 3600

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent                   # llm_CGCC/api_llm_CGCC/
CGCC_DIR = SCRIPT_DIR.parent / "CGCC"               # llm_CGCC/CGCC/
PYTHON_EXEC = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"  # torch works here
PROMPT_PATH = SCRIPT_DIR / "prompt_filled.md"
RUN_DIR = SCRIPT_DIR / "run_llama_api"


def extract_block(text: str, tag: str) -> str:
    """
    Extract content from <tag>...</tag>.
    Falls back to fenced code blocks if tags are missing
    (some models skip XML tags and wrap code in ``` instead).
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
        f"working_directory: {CGCC_DIR}",
    ]
    pip = subprocess.run(
        ["python", "-m", "pip", "freeze"],
        capture_output=True, text=True, check=False,
    )
    lines += ["\n[pip freeze]", pip.stdout]
    (run_dir / "environment.txt").write_text("\n".join(lines))


def validate_submission(cgcc_dir: Path) -> str:
    """
    Check that submission.csv was written correctly.
    CGCC requires: filename, prediction columns, 36 rows (one per test graph).
    """
    csv_path = cgcc_dir / "submission.csv"
    if not csv_path.exists():
        return "FAIL: submission.csv not found"

    lines = csv_path.read_text().strip().splitlines()
    if not lines:
        return "FAIL: submission.csv is empty"

    header = lines[0].strip()
    if header != "filename,prediction":
        return f"FAIL: wrong header '{header}', expected 'filename,prediction'"

    n_rows = len(lines) - 1  # exclude header
    if n_rows != 36:
        return f"FAIL: expected 36 prediction rows, got {n_rows}"

    return f"OK: submission.csv has {n_rows} rows with correct header"


def call_llm(client: Together, messages: list) -> str:
    """Single API call to Together AI. Follows ineqmath's together.py pattern."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return response.choices[0].message.content or ""


def main():
    if "TOGETHER_API_KEY" not in os.environ:
        raise RuntimeError(
            "TOGETHER_API_KEY is not set.\n"
            "Sign up free at https://api.together.ai and export your key:\n"
            "  export TOGETHER_API_KEY=your_key_here"
        )

    if not CGCC_DIR.exists():
        raise FileNotFoundError(f"CGCC competition folder not found at: {CGCC_DIR}")

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"prompt_filled.md not found at: {PROMPT_PATH}")

    RUN_DIR.mkdir(exist_ok=True)

    # Save config for reproducibility
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
        "cgcc_dir": str(CGCC_DIR),
    }
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))
    write_environment(RUN_DIR)

    # Copy prompt snapshot into run folder
    prompt_text = PROMPT_PATH.read_text()
    (RUN_DIR / "prompt_filled.md").write_text(prompt_text)

    # Build initial message list (user-only, same as groq_api_mode.py pattern)
    messages = [{"role": "user", "content": prompt_text}]

    client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    success = False
    repairs_used = 0
    total_start = time.time()

    for attempt in range(REPAIR_LIMIT + 1):
        print(f"\n=== Together AI attempt {attempt} | model={MODEL} ===")

        # --- Call LLM ---
        text = call_llm(client, messages)
        (RUN_DIR / f"attempt_{attempt}_raw_response.txt").write_text(text)

        # --- Parse plan and code ---
        try:
            plan = extract_block(text, "plan")
            code = extract_block(text, "code")
        except Exception as e:
            parse_error = str(e)
            print(f"Parse error: {parse_error}")
            (RUN_DIR / f"attempt_{attempt}_parse_error.txt").write_text(parse_error)

            # Ask LLM to re-format
            messages.append({"role": "assistant", "content": text})
            messages.append({
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
        (RUN_DIR / "plan.md").write_text(plan)
        (RUN_DIR / "solution.py").write_text(code)

        # Write solution.py into CGCC root so `python solution.py` works
        solution_in_cgcc = CGCC_DIR / "solution.py"
        solution_in_cgcc.write_text(code)

        # --- Execute solution.py from CGCC root ---
        print(f"Running solution.py from {CGCC_DIR} ...")
        start = time.time()
        result = subprocess.run(
            [PYTHON_EXEC, "solution.py"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
            check=False,
            cwd=str(CGCC_DIR),              # run from competition repo root
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
        print(execution_log[:500])

        # --- Validate submission.csv ---
        validation_log = validate_submission(CGCC_DIR)
        (RUN_DIR / f"attempt_{attempt}_validation_log.txt").write_text(validation_log)
        (RUN_DIR / "validation_log.txt").write_text(validation_log)
        print(f"Validation: {validation_log}")

        # --- Success? ---
        if result.returncode == 0 and validation_log.startswith("OK"):
            print("Success: submission.csv written correctly.")
            shutil.copyfile(CGCC_DIR / "submission.csv", RUN_DIR / "submission.csv")
            shutil.copyfile(solution_in_cgcc, RUN_DIR / "final_solution.py")
            success = True
            repairs_used = attempt
            break

        # --- Repair: send traceback back ---
        if attempt < REPAIR_LIMIT:
            messages.append({"role": "assistant", "content": text})
            messages.append({
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
