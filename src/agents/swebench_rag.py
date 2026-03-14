"""
RAG (Retrieval-Augmented Generation) module for SWE-bench.

Retrieves relevant source files from the actual repository at the correct
commit, then injects them into the prompt so the model can generate patches
that reference real file paths and real code.

Strategy:
  1. Clone/checkout the repo at the base commit specified by the instance.
  2. Parse the problem_statement to extract likely file paths and keywords.
  3. Search the repo for relevant files (keyword grep + path matching).
  4. Rank files by relevance and select the top-k (fitting within token budget).
  5. Build a prompt with the file contents injected.

Usage:
    from src.agents.swebench_rag import SWEBenchRAG

    rag = SWEBenchRAG(cache_dir="/tmp/swebench_repos")
    context = rag.retrieve(instance)
    # context = {"files": [{"path": "django/core/validators.py", "content": "..."}], ...}
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

# -----------------------------------------------------------------------
# Repo cache
# -----------------------------------------------------------------------
DEFAULT_CACHE_DIR = Path("/tmp/swebench_repos")

# File extensions we care about
CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".rb", ".go", ".rs", ".c", ".cpp", ".h"}

# Max characters per file to include (truncate long files)
MAX_FILE_CHARS = 3000

# Max total context characters (all files combined)
MAX_CONTEXT_CHARS = 8000


class SWEBenchRAG:
    """Retrieves relevant source files from SWE-bench repositories."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR, max_files: int = 5,
                 max_context_chars: int = MAX_CONTEXT_CHARS):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_files = max_files
        self.max_context_chars = max_context_chars

    # ------------------------------------------------------------------
    # Repo checkout
    # ------------------------------------------------------------------
    def _get_repo_dir(self, repo: str) -> Path:
        """Get the local directory for a repo (e.g., django/django -> django__django)."""
        return self.cache_dir / repo.replace("/", "__")

    def checkout(self, instance: dict) -> Path:
        """Clone the repo (if needed) and checkout the base commit.

        Returns the path to the repo directory at the correct commit.
        """
        repo = instance["repo"]
        base_commit = instance["base_commit"]
        repo_dir = self._get_repo_dir(repo)

        if not (repo_dir / ".git").exists():
            print(f"    [RAG] Cloning {repo}...")
            subprocess.run(
                ["git", "clone", "--quiet", f"https://github.com/{repo}.git",
                 str(repo_dir)],
                check=True, capture_output=True, timeout=120,
            )
        else:
            # Fetch to ensure we have the commit
            subprocess.run(
                ["git", "fetch", "--quiet", "--all"],
                cwd=repo_dir, capture_output=True, timeout=60,
            )

        # Checkout the base commit
        subprocess.run(
            ["git", "checkout", "--quiet", "--force", base_commit],
            cwd=repo_dir, check=True, capture_output=True, timeout=30,
        )
        # Clean untracked files
        subprocess.run(
            ["git", "clean", "-fdx", "--quiet"],
            cwd=repo_dir, capture_output=True, timeout=30,
        )

        return repo_dir

    # ------------------------------------------------------------------
    # Keyword extraction from problem statement
    # ------------------------------------------------------------------
    @staticmethod
    def extract_keywords(problem_statement: str) -> list[str]:
        """Extract likely file paths, class names, and function names from
        the problem statement."""
        keywords = []

        # 1. Explicit file paths (e.g., django/core/validators.py)
        file_paths = re.findall(r'[\w/]+\.py\b', problem_statement)
        keywords.extend(file_paths)

        # 2. Python dotted paths (e.g., django.core.validators)
        dotted = re.findall(r'[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*){2,}', problem_statement)
        for d in dotted:
            # Convert dotted path to file path guess
            parts = d.split(".")
            keywords.append("/".join(parts[:-1]) + ".py")  # module path
            keywords.append(parts[-1])  # last component (class/func name)

        # 3. Class names (CamelCase)
        classes = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', problem_statement)
        keywords.extend(classes)

        # 4. Function/method names mentioned with ()
        funcs = re.findall(r'\b([a-z_]\w+)\s*\(', problem_statement)
        keywords.extend(f for f in funcs if len(f) > 3 and f not in {
            "with", "from", "that", "this", "when", "have", "been", "does",
            "should", "would", "could", "like", "some", "into", "also",
        })

        # 5. Error class names
        errors = re.findall(r'\b\w+Error\b|\b\w+Exception\b|\b\w+Warning\b',
                            problem_statement)
        keywords.extend(errors)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for kw in keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                unique.append(kw)

        return unique

    # ------------------------------------------------------------------
    # File search
    # ------------------------------------------------------------------
    def search_files(self, repo_dir: Path, keywords: list[str]) -> list[dict]:
        """Search the repo for files relevant to the keywords.

        Returns a list of {"path": relative_path, "score": int} sorted by score.
        """
        # Build a set of all Python files in the repo
        all_files = []
        for ext in CODE_EXTENSIONS:
            for f in repo_dir.rglob(f"*{ext}"):
                rel = f.relative_to(repo_dir)
                # Skip hidden dirs, venvs, tests (lower priority), migrations
                rel_str = str(rel)
                if any(p.startswith(".") for p in rel.parts):
                    continue
                if any(p in ("venv", "node_modules", "__pycache__", ".tox")
                       for p in rel.parts):
                    continue
                all_files.append({"path": rel_str, "abs_path": str(f), "score": 0})

        if not all_files:
            return []

        # Score files by keyword matches
        for f in all_files:
            path_lower = f["path"].lower()
            for kw in keywords:
                kw_lower = kw.lower()
                # Path contains keyword (strongest signal)
                if kw_lower in path_lower:
                    f["score"] += 10
                # Path contains keyword as a path component
                kw_as_path = kw_lower.replace(".", "/")
                if kw_as_path in path_lower:
                    f["score"] += 15

        # For files with score > 0, also grep for keyword mentions in content
        scored_files = [f for f in all_files if f["score"] > 0]

        # If no path matches, do a brute-force grep on top keywords
        if not scored_files and keywords:
            top_keywords = keywords[:5]
            for f in all_files:
                try:
                    content = Path(f["abs_path"]).read_text(errors="ignore")
                    for kw in top_keywords:
                        if kw in content:
                            f["score"] += 3
                except Exception:
                    pass
            scored_files = [f for f in all_files if f["score"] > 0]

        # Boost non-test files
        for f in scored_files:
            if "test" in f["path"].lower():
                f["score"] -= 2  # slightly deprioritize tests
            if "__init__" in f["path"]:
                f["score"] -= 1  # slightly deprioritize init files

        # Sort by score descending
        scored_files.sort(key=lambda x: x["score"], reverse=True)
        return scored_files

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------
    def build_context(self, repo_dir: Path, files: list[dict]) -> list[dict]:
        """Read file contents for the top-k files, respecting token budget.

        Returns list of {"path": str, "content": str}.
        """
        context_files = []
        total_chars = 0

        for f in files[:self.max_files * 2]:  # check more than needed in case some are too large
            if len(context_files) >= self.max_files:
                break
            if total_chars >= self.max_context_chars:
                break

            try:
                abs_path = repo_dir / f["path"]
                content = abs_path.read_text(errors="ignore")

                # Truncate long files
                if len(content) > MAX_FILE_CHARS:
                    content = content[:MAX_FILE_CHARS] + "\n... (truncated)"

                # Check budget
                if total_chars + len(content) > self.max_context_chars:
                    # Include partial if we have room for at least 500 chars
                    remaining = self.max_context_chars - total_chars
                    if remaining > 500:
                        content = content[:remaining] + "\n... (truncated)"
                    else:
                        continue

                context_files.append({"path": f["path"], "content": content})
                total_chars += len(content)

            except Exception:
                continue

        return context_files

    # ------------------------------------------------------------------
    # Main retrieve method
    # ------------------------------------------------------------------
    def retrieve(self, instance: dict) -> dict:
        """Full RAG pipeline for one SWE-bench instance.

        Returns:
            {
                "files": [{"path": str, "content": str}, ...],
                "keywords": [str, ...],
                "n_candidates": int,
                "repo_dir": str,
            }
        """
        repo_dir = self.checkout(instance)

        keywords = self.extract_keywords(instance["problem_statement"])
        print(f"    [RAG] Keywords: {keywords[:8]}")

        candidates = self.search_files(repo_dir, keywords)
        print(f"    [RAG] Found {len(candidates)} candidate files")

        context_files = self.build_context(repo_dir, candidates)
        print(f"    [RAG] Selected {len(context_files)} files "
              f"({sum(len(f['content']) for f in context_files)} chars)")

        return {
            "files": context_files,
            "keywords": keywords,
            "n_candidates": len(candidates),
            "repo_dir": str(repo_dir),
        }

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------
    @staticmethod
    def format_context_for_prompt(context: dict) -> str:
        """Format retrieved files into a string suitable for injection into a prompt."""
        if not context["files"]:
            return "(No relevant source files found)"

        parts = []
        for f in context["files"]:
            parts.append(f"=== {f['path']} ===\n{f['content']}")
        return "\n\n".join(parts)


# -----------------------------------------------------------------------
# Prompt templates with RAG
# -----------------------------------------------------------------------
RAG_SINGLE_STEP_PROMPT = (
    "You are a software engineer fixing a bug in {repo}. "
    "Given the problem description and the relevant source code, "
    "generate a unified diff patch that fixes the issue.\n\n"
    "Problem:\n{problem_statement}\n\n"
    "Relevant source files:\n{context}\n\n"
    "Generate ONLY a unified diff (git diff format). "
    "Use the EXACT file paths shown above. "
    "Start with ```diff and end with ```. Example format:\n"
    "```diff\n"
    "diff --git a/path/to/file.py b/path/to/file.py\n"
    "--- a/path/to/file.py\n"
    "+++ b/path/to/file.py\n"
    "@@ -10,3 +10,4 @@\n"
    " existing line\n"
    "-old line\n"
    "+new line\n"
    "```"
)

RAG_DYNAMIC_STEP1_PROMPT = (
    "Analyze this bug report for {repo} and identify which files and functions "
    "are involved. Here is the relevant source code:\n\n"
    "{context}\n\n"
    "Bug report:\n{problem_statement}\n\n"
    "Identify the root cause — which file, which function, which line."
)

RAG_DYNAMIC_STEP2_PROMPT = (
    "Based on this analysis:\n{step1_output}\n\n"
    "And this source code:\n{context_summary}\n\n"
    "What exact change is needed to fix this bug? Be specific about "
    "the file path, the line to change, and the fix."
)

RAG_DYNAMIC_STEP3_PROMPT = (
    "Based on this diagnosis:\n{step2_output}\n\n"
    "Generate a unified diff patch using the exact file paths from the repository. "
    "Start with ```diff and end with ```."
)


# -----------------------------------------------------------------------
# Quick test
# -----------------------------------------------------------------------
def main():
    """Quick test: retrieve context for the first SWE-bench instance."""
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test",
                      trust_remote_code=True)

    # Pick a django instance
    instance = None
    for row in ds:
        if row["repo"] == "django/django":
            instance = dict(row)
            break

    if instance is None:
        print("No django instance found")
        return

    print(f"Instance: {instance['instance_id']}")
    print(f"Base commit: {instance['base_commit']}")
    print(f"Problem (first 200 chars): {instance['problem_statement'][:200]}")
    print()

    rag = SWEBenchRAG()
    context = rag.retrieve(instance)

    print(f"\nRetrieved {len(context['files'])} files:")
    for f in context["files"]:
        print(f"  {f['path']} ({len(f['content'])} chars)")

    print(f"\nFormatted prompt context (first 500 chars):")
    formatted = SWEBenchRAG.format_context_for_prompt(context)
    print(formatted[:500])


if __name__ == "__main__":
    main()
