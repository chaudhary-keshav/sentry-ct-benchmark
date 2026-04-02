#!/usr/bin/env python3
"""
CodeTrellis Chunked Agentic PR Reviewer (v3 — adapted for any codebase)
========================================================================
Architecture-aware PR reviewer using CodeTrellis context engine.
Uses a chunked multi-pass design to work within any OpenAI TPM limit
while using the strongest available model.

  Phase 1: GATHER (Python only — no LLM calls)
  ├── git diff → changed files + per-file diffs
  ├── codetrellis export → project overview, best practices
  └── codetrellis context → per-file dependency context

  Phase 2: REVIEW (per-chunk LLM + investigation tools)
  ├── 3 files per chunk
  ├── 5 agentic rounds per chunk
  ├── 65s TPM cooldown between chunks
  └── 3 investigation tools

  Phase 3: SYNTHESIZE (single LLM call)
  └── Merge, deduplicate, cross-file analysis → final review

Usage:
  OPENAI_API_KEY=sk-... python3 scripts/codetrellis-pr-reviewer.py

Environment Variables:
  OPENAI_API_KEY        - OpenAI API key (required)
  GITHUB_TOKEN          - GitHub token (for posting PR review comments)
  REPO                  - GitHub repo (owner/name)
  PR_NUMBER             - PR number to review
  MODEL                 - Override model (auto-detects best available if empty)
  MAX_ROUNDS_PER_CHUNK  - Max agentic rounds per file chunk (default: 5)
  CHUNK_SIZE            - Files per chunk (default: 3)
  INTER_CHUNK_DELAY     - Seconds between API calls for TPM cooldown (default: 65)
"""

import json
import os
import subprocess
import sys
import urllib.request
import re
import time as _time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_ROUNDS_PER_CHUNK = int(os.environ.get("MAX_ROUNDS_PER_CHUNK", "5"))
MAX_FILE_CHARS = 8000
MAX_CONTEXT_CHARS = 10000
MAX_SECTION_CHARS = 6000
MAX_SEARCH_RESULTS = 30
MAX_TOKENS = 4096
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "3"))
INTER_CHUNK_DELAY = int(os.environ.get("INTER_CHUNK_DELAY", "65"))
PR_NUMBER = os.environ.get("PR_NUMBER", "")
REPO = os.environ.get("REPO", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("MODEL", "")
API_URL = "https://api.openai.com/v1/chat/completions"
MODEL_PRIORITY = [
    "gpt-5.2", "gpt-5.1-codex-max", "gpt-4.1", "gpt-4o",
    "o3-mini", "gpt-4.1-mini", "gpt-4o-mini",
]

# ---------------------------------------------------------------------------
# Tool Definitions — Investigation tools (language-agnostic)
# ---------------------------------------------------------------------------
INVESTIGATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "Search the codebase for a pattern using grep. Returns matching lines "
                "with file paths. Use to find: existing implementations, duplicate patterns, "
                "usages of classes/functions, imports/exports, configuration. "
                "Searches .py .ts .js .json .yaml .yml .toml .cfg files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (grep regex)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory scope (default: '.')",
                        "default": ".",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read raw contents of a file NOT in the changed set. "
                "Use to inspect related services, existing implementations, tests, "
                "or configs you need to compare against."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from repo root",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_additional_context",
            "description": (
                "Fetch a CodeTrellis matrix section for the project. "
                "Available sections include: OVERVIEW, BEST_PRACTICES, "
                "IMPLEMENTATION_LOGIC, ERROR_HANDLING, INFRASTRUCTURE, "
                "BUSINESS_DOMAIN, PYTHON_TYPES, PYTHON_API, PYTHON_FUNCTIONS, "
                "and others. Use to understand project-wide conventions, "
                "coding standards, and architectural patterns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section_name": {
                        "type": "string",
                        "description": "Matrix section name (e.g. BEST_PRACTICES, ERROR_HANDLING)",
                    }
                },
                "required": ["section_name"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def tool_read_file(path: str) -> str:
    safe_path = os.path.normpath(path)
    if safe_path.startswith("..") or os.path.isabs(safe_path):
        return f"Error: Invalid path '{path}'"
    if not os.path.isfile(safe_path):
        return f"Error: File not found: {path}"
    try:
        with open(safe_path, "r", errors="replace") as f:
            content = f.read(MAX_FILE_CHARS)
        if len(content) >= MAX_FILE_CHARS:
            content += f"\n... [truncated at {MAX_FILE_CHARS} chars]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def tool_search_code(pattern: str, path: str = ".") -> str:
    safe_path = os.path.normpath(path)
    if safe_path.startswith("..") or os.path.isabs(safe_path):
        return f"Error: Invalid path '{path}'"
    try:
        result = subprocess.run(
            [
                "grep", "-rn",
                "--include=*.py", "--include=*.ts", "--include=*.js",
                "--include=*.json", "--include=*.yaml", "--include=*.yml",
                "--include=*.toml", "--include=*.cfg",
                pattern, safe_path,
            ],
            capture_output=True, text=True, timeout=15,
        )
        output = result.stdout or "(no matches)"
        lines = output.strip().split("\n")
        if len(lines) > MAX_SEARCH_RESULTS:
            output = (
                "\n".join(lines[:MAX_SEARCH_RESULTS])
                + f"\n... [{len(lines) - MAX_SEARCH_RESULTS} more]"
            )
        return output
    except subprocess.TimeoutExpired:
        return "Error: Search timed out"
    except Exception as e:
        return f"Error: {e}"


def tool_get_additional_context(section_name: str) -> str:
    """Fetch a CodeTrellis matrix section."""
    clean = re.sub(r"[^A-Za-z0-9_]", "", section_name)
    try:
        proc = subprocess.run(
            ["codetrellis", "export", ".", "--section", clean],
            capture_output=True, text=True, timeout=30,
        )
        output = (proc.stdout or "").strip()
        if not output:
            return f"Section '{clean}' not found or empty."
        return output[:MAX_SECTION_CHARS]
    except Exception as e:
        return f"Error fetching section: {e}"


TOOL_HANDLERS = {
    "read_file": lambda inp: tool_read_file(inp["path"]),
    "search_code": lambda inp: tool_search_code(
        inp["pattern"], inp.get("path", ".")
    ),
    "get_additional_context": lambda inp: tool_get_additional_context(
        inp["section_name"]
    ),
}

# ---------------------------------------------------------------------------
# Model Auto-Detection
# ---------------------------------------------------------------------------

def detect_best_model() -> str:
    """Query OpenAI API for available models, pick the strongest one."""
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        available = {m["id"] for m in data.get("data", [])}
        gpt_models = sorted(m for m in available if "gpt" in m or "o3" in m or "o4" in m)
        print(f"[Model] Available: {gpt_models[:15]}")

        for model in MODEL_PRIORITY:
            if model in available:
                print(f"[Model] Selected: {model}")
                return model
    except Exception as e:
        print(f"[Model] Detection failed ({e}), using gpt-4o")
    return "gpt-4o"


# ---------------------------------------------------------------------------
# Context Pre-Gathering (no LLM calls — all local)
# ---------------------------------------------------------------------------

def get_changed_files_list() -> list:
    """Get list of changed file paths from git."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "origin/master..HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except Exception:
        return []


def extract_file_diff(full_diff: str, filepath: str) -> str:
    """Extract the diff chunk for a specific file from the full PR diff."""
    parts = re.split(r"(?=^diff --git )", full_diff, flags=re.MULTILINE)
    for part in parts:
        first_line = part.split("\n")[0] if part.strip() else ""
        if f"a/{filepath}" in first_line or f"b/{filepath}" in first_line:
            return part[:MAX_FILE_CHARS]
    return ""


def pre_gather_context(changed_files: list) -> dict:
    """Pre-gather ALL CodeTrellis context locally. No LLM calls needed."""
    ctx = {"overview": "", "best_practices": "", "file_contexts": {}}

    # Project overview
    try:
        proc = subprocess.run(
            ["codetrellis", "export", ".", "--section", "OVERVIEW"],
            capture_output=True, text=True, timeout=30,
        )
        ctx["overview"] = (proc.stdout or "")[:MAX_SECTION_CHARS]
        print(f"[Context] Overview: {len(ctx['overview'])} chars")
    except Exception as e:
        print(f"[Context] Overview failed: {e}")

    # Best practices (pre-gathered — not left to LLM discovery)
    try:
        proc = subprocess.run(
            ["codetrellis", "export", ".", "--section", "BEST_PRACTICES"],
            capture_output=True, text=True, timeout=30,
        )
        ctx["best_practices"] = (proc.stdout or "")[:MAX_SECTION_CHARS]
        print(f"[Context] Best Practices: {len(ctx['best_practices'])} chars")
    except Exception as e:
        print(f"[Context] Best Practices failed: {e}")

    # Per-file CodeTrellis context
    for fp in changed_files:
        safe = os.path.normpath(fp)
        if safe.startswith("..") or os.path.isabs(safe):
            continue
        try:
            proc = subprocess.run(
                ["codetrellis", "context", safe, "--project", "."],
                capture_output=True, text=True, timeout=30,
            )
            ctx["file_contexts"][fp] = (proc.stdout or "")[:MAX_CONTEXT_CHARS]
            print(f"[Context] {fp}: {len(ctx['file_contexts'][fp])} chars")
        except Exception:
            ctx["file_contexts"][fp] = "(context unavailable)"

    return ctx


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CHUNK_SYSTEM_PROMPT = """You are a principal software engineer performing a thorough architecture-aware PR review.

## YOUR CONTEXT (already provided — do NOT ask for more context)
You have been given:
1. **Project overview** — architecture, tech stack, patterns
2. **Best practices** — coding standards, conventions for the project
3. **Per-file CodeTrellis context** — each file's full dependency tree, related types, imports
4. **The PR diff** for each file in this chunk

## INVESTIGATION TOOLS
You have 3 tools for deeper investigation. USE THEM AGGRESSIVELY before making findings:

- `search_code(pattern)` — Find existing patterns. SEARCH BEFORE CLAIMING something is missing:
  - Error handling: `search_code("raise\\|except\\|try:")`
  - Authentication: `search_code("auth\\|permission\\|guard")`
  - Feature flags: `search_code("features.has\\|feature_flag")`
  - Related code: `search_code("class_name\\|function_name")`

- `read_file(path)` — Read related files to compare patterns:
  - Other modules: How do they handle similar operations?
  - Tests: Is the changed code path tested?
  - Config: What settings affect this code?

- `get_additional_context(section_name)` — Fetch CodeTrellis matrix sections:
  - `get_additional_context("ERROR_HANDLING")` — Project error patterns
  - `get_additional_context("IMPLEMENTATION_LOGIC")` — Core implementation patterns
  - `get_additional_context("INFRASTRUCTURE")` — Deployment/infra patterns
  - `get_additional_context("PYTHON_TYPES")` — Type definitions
  - `get_additional_context("PYTHON_API")` — API endpoints
  - `get_additional_context("BUSINESS_DOMAIN")` — Domain logic

## INVESTIGATION STRATEGY (MANDATORY)
Your first response MUST be a tool call. Before reporting ANY findings:
1. Read the full source files (not just the diff) for all changed files
2. Search for related patterns across the codebase
3. Fetch at least 2-3 matrix sections for project-wide context
4. THEN report findings with evidence from your investigation

## FINDING CATEGORIES
1. **CRITICAL** — Security (auth bypass, injection), bugs (race conditions, null refs), data loss
2. **ARCHITECTURE** — Duplicate systems, missing integration with existing modules, pattern violations
3. **IMPORTANT** — N+1 queries, missing validation, performance issues, deployment concerns
4. **SUGGESTION** — Code quality, test coverage, naming

## WHAT MAKES A GREAT FINDING
1. **VERIFIED**: You searched for existing patterns before claiming something is wrong
2. **SPECIFIC**: Exact file path and line number
3. **ACTIONABLE**: Clear recommendation with code example if helpful
4. **CONTEXTUAL**: Explains WHY it matters in context of the project's architecture

Output each finding as:
**[CATEGORY]** [Title] — `file:line`
[Problem description]
**Recommendation:** [Specific fix]"""


SYNTHESIS_PROMPT = """You are consolidating findings from a chunked PR review into one final review. Below are findings from separate file-group reviews.

Your tasks:
1. **Deduplicate** — merge identical or overlapping findings
2. **Cross-file analysis** — identify issues spanning multiple file groups that individual chunks couldn't see
3. **Prioritize** — ensure CRITICAL findings are accurate and well-supported

Output format:

## 🔍 PR Review: [Brief Title]

### Summary
[2-3 sentences: what the PR does, overall assessment]

### Findings

#### 🔴 CRITICAL
1. **[Title]** — `file:line`
   [Problem] → **Recommendation:** [Fix]

#### 🟠 ARCHITECTURE
1. ...

#### 🟡 IMPORTANT
1. ...

#### 💡 SUGGESTIONS
1. ...

### Verdict: [APPROVE | REQUEST_CHANGES | COMMENT]
[Brief justification]

Be thorough. Keep ALL valid unique findings. Remove true duplicates only."""


# ---------------------------------------------------------------------------
# OpenAI API Caller
# ---------------------------------------------------------------------------

_resolved_model = ""


def api_call(messages: list, tools: list = None, tool_choice: str = "auto") -> tuple:
    """Call OpenAI API. Returns (tool_calls, text, raw_message)."""
    global _resolved_model
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    model = _resolved_model or MODEL or "gpt-4o"

    token_key = (
        "max_completion_tokens"
        if model.startswith(("gpt-5", "o3", "o4"))
        else "max_tokens"
    )

    payload = {
        "model": model,
        "messages": messages,
        token_key: MAX_TOKENS,
        "temperature": 0.1,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    data = json.dumps(payload).encode()
    max_retries = 5
    for attempt in range(max_retries):
        req = urllib.request.Request(
            API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read())
            choice = result["choices"][0]
            msg = choice["message"]
            tool_calls = msg.get("tool_calls", [])
            text = msg.get("content", "") or ""
            usage = result.get("usage", {})
            if usage:
                t_in = usage.get("prompt_tokens", 0)
                t_out = usage.get("completion_tokens", 0)
                print(f"  [Tokens] in={t_in} out={t_out} total={t_in + t_out}")
            return tool_calls, text, msg
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            if e.code == 429 and attempt < max_retries - 1:
                wait = min(20 * (2**attempt), 120)
                print(
                    f"  [429] Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})"
                )
                _time.sleep(wait)
            else:
                print(f"  [HTTP {e.code}] {body[:500]}")
                raise


# ---------------------------------------------------------------------------
# Chunked Review Engine
# ---------------------------------------------------------------------------

def review_chunk(
    chunk_files: list, full_diff: str, ctx: dict, all_files: list
) -> str:
    """Run a mini agentic review on a chunk of 2-3 files."""
    file_sections = []
    for fp in chunk_files:
        diff_chunk = extract_file_diff(full_diff, fp)
        file_ctx = ctx["file_contexts"].get(fp, "")
        file_sections.append(
            f"### File: {fp}\n"
            f"**CodeTrellis Context (dependencies, types, imports):**\n{file_ctx}\n\n"
            f"**Diff:**\n```diff\n{diff_chunk}\n```\n"
        )

    user_content = (
        f"Review these {len(chunk_files)} files from a PR with {len(all_files)} "
        f"total changed files.\n"
        f"All changed files in this PR: {', '.join(all_files)}\n\n"
        f"**Project Overview:**\n{ctx['overview']}\n\n"
        f"**Best Practices:**\n{ctx['best_practices']}\n\n"
        + "\n".join(file_sections)
        + "\n\nIMPORTANT: You MUST use investigation tools first. Start by calling "
        "read_file on the changed files to see the FULL source (not just the diff). "
        "Then use search_code and get_additional_context to gather project-wide "
        "patterns. Only THEN provide your detailed findings."
    )

    messages = [
        {"role": "system", "content": CHUNK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for round_num in range(1, MAX_ROUNDS_PER_CHUNK + 1):
        print(f"    Round {round_num}/{MAX_ROUNDS_PER_CHUNK}...")

        # Force tool use on round 1, auto after that
        choice = "required" if round_num == 1 else "auto"
        tool_calls, text, raw_msg = api_call(
            messages, INVESTIGATION_TOOLS, tool_choice=choice
        )

        # No tool calls means the model produced its findings
        if not tool_calls:
            return text

        # Process investigation tool calls
        messages.append(raw_msg)
        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            try:
                inp = json.loads(fn["arguments"])
            except json.JSONDecodeError:
                inp = {}
            tid = tc["id"]

            print(f"      [Tool] {name}({json.dumps(inp)[:80]})")
            handler = TOOL_HANDLERS.get(name)
            result = handler(inp) if handler else f"Unknown tool: {name}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": result[:MAX_FILE_CHARS],
                }
            )

    # Force findings if max rounds hit
    messages.append(
        {
            "role": "user",
            "content": "Produce your findings now based on what you've gathered.",
        }
    )
    _, text, _ = api_call(messages)
    return text


def synthesize_findings(chunk_findings: list) -> str:
    """Merge findings from all chunks into final consolidated review."""
    findings_text = ""
    for i, (files, findings) in enumerate(chunk_findings):
        findings_text += (
            f"\n---\n**Chunk {i + 1}** (files: {', '.join(files)}):\n{findings}\n"
        )

    messages = [
        {"role": "system", "content": SYNTHESIS_PROMPT},
        {
            "role": "user",
            "content": (
                f"Merge these findings from {len(chunk_findings)} review chunks into one "
                f"consolidated review:\n{findings_text}"
            ),
        },
    ]

    print("[Synthesis] Merging all findings...")
    _, text, _ = api_call(messages)
    return text


def run_chunked_review(pr_diff: str) -> str:
    """Main review orchestrator: gather → chunk → review → synthesize."""
    global _resolved_model

    # 1. Detect best model
    if MODEL:
        _resolved_model = MODEL
        print(f"[Model] Using configured: {_resolved_model}")
    else:
        _resolved_model = detect_best_model()

    # 2. Get changed files
    changed_files = get_changed_files_list()
    if not changed_files:
        return "No changed files found."
    print(f"[Gather] {len(changed_files)} changed files: {', '.join(changed_files)}")

    # 3. Pre-gather ALL context locally (no LLM calls)
    print("[Gather] Pre-gathering CodeTrellis context...")
    start = _time.time()
    ctx = pre_gather_context(changed_files)
    gather_time = _time.time() - start
    print(f"[Gather] Done in {gather_time:.1f}s")

    # 4. Chunk files into TPM-safe groups
    chunks = [
        changed_files[i : i + CHUNK_SIZE]
        for i in range(0, len(changed_files), CHUNK_SIZE)
    ]
    print(
        f"[Review] {len(chunks)} chunk(s) of <={CHUNK_SIZE} files | Model: {_resolved_model}"
    )

    # 5. Review each chunk with mini agentic loop
    chunk_findings = []
    review_start = _time.time()

    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i + 1}/{len(chunks)}] Files: {', '.join(chunk)}")

        findings = review_chunk(chunk, pr_diff, ctx, changed_files)
        chunk_findings.append((chunk, findings))
        print(f"[Chunk {i + 1}] {len(findings)} chars of findings")

        if i < len(chunks) - 1:
            print(f"[Rate] Waiting {INTER_CHUNK_DELAY}s for TPM cooldown...")
            _time.sleep(INTER_CHUNK_DELAY)

    # 6. Synthesize or return directly
    if len(chunk_findings) == 1:
        final = chunk_findings[0][1]
    else:
        print(f"[Rate] Waiting {INTER_CHUNK_DELAY}s before synthesis...")
        _time.sleep(INTER_CHUNK_DELAY)
        final = synthesize_findings(chunk_findings)

    total_time = _time.time() - review_start
    print(f"\n[Agent] Review complete: {total_time:.1f}s total, {len(chunks)} chunks")
    return final


# ---------------------------------------------------------------------------
# Post Review to GitHub
# ---------------------------------------------------------------------------

def post_review_comment(body: str):
    """Post review as a PR comment."""
    if not PR_NUMBER or not GITHUB_TOKEN or not REPO:
        print("[Post] Missing PR_NUMBER, GITHUB_TOKEN, or REPO — printing only")
        return

    api_url = f"https://api.github.com/repos/{REPO}/issues/{PR_NUMBER}/comments"
    _cleanup_old_comments()

    payload = json.dumps({"body": body}).encode()
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            print("[Post] Review posted to GitHub successfully")
    except Exception as e:
        print(f"[Post] Failed: {e}")


def _cleanup_old_comments():
    """Remove previous CodeTrellis v3 script review comments from this PR."""
    api_url = f"https://api.github.com/repos/{REPO}/issues/{PR_NUMBER}/comments"
    req = urllib.request.Request(
        api_url,
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            comments = json.loads(resp.read())
        for c in comments:
            body = c.get("body") or ""
            user = c.get("user", {}).get("login", "")
            if "CodeTrellis AI PR Review (v3" in body and (
                "github-actions" in user
                or c.get("user", {}).get("type") == "Bot"
            ):
                del_url = (
                    f"https://api.github.com/repos/{REPO}/issues/comments/{c['id']}"
                )
                del_req = urllib.request.Request(
                    del_url,
                    method="DELETE",
                    headers={
                        "Authorization": f"Bearer {GITHUB_TOKEN}",
                        "Accept": "application/vnd.github+json",
                    },
                )
                urllib.request.urlopen(del_req, timeout=10)
                print(f"[Post] Cleaned up old v3 script comment {c['id']}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not set — skipping AI review")
        sys.exit(0)

    # Get the PR diff
    diff_file = os.environ.get("DIFF_FILE", "/tmp/pr_diff.txt")
    if os.path.isfile(diff_file):
        with open(diff_file) as f:
            pr_diff = f.read()
    else:
        result = subprocess.run(
            ["git", "diff", "origin/master..HEAD"],
            capture_output=True, text=True,
        )
        pr_diff = result.stdout

    if not pr_diff.strip():
        print("No diff found — nothing to review")
        sys.exit(0)

    print(f"[Agent] Diff: {len(pr_diff)} chars")

    # Run chunked review
    review = run_chunked_review(pr_diff)

    # Output
    print("\n" + "=" * 60)
    print("REVIEW")
    print("=" * 60)
    print(review)

    # Post to GitHub
    model_name = _resolved_model or MODEL or "unknown"
    full_body = (
        f"## 🔍 CodeTrellis AI PR Review (v3 Script — Chunked Architecture)\n\n"
        f"{review}\n\n"
        f"---\n"
        f"*Powered by {model_name} + CodeTrellis context engine "
        f"(v3 chunked architecture with pre-gathered context + investigation tools)*"
    )
    post_review_comment(full_body)


if __name__ == "__main__":
    main()
