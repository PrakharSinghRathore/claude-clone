"""
Smart Git Manager with conventional-commit message generation.

Provides an async wrapper around common git operations, parsing structured
output so callers receive clean Python data structures instead of raw text.

Usage:
    gm = GitManager("/path/to/repo")
    st = await gm.status()
    result = await gm.commit()
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GitManager:
    """Async git operations manager."""

    def __init__(self, repo_path: str = ".") -> None:
        self.repo_path = os.path.abspath(repo_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_git(
        self,
        *args: str,
        input_data: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> tuple[int, str, str]:
        """Execute a git command and return (returncode, stdout, stderr)."""
        cmd = ("git",) + args
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if input_data else asyncio.subprocess.DEVNULL,
            cwd=cwd or self.repo_path,
        )
        stdin_bytes = input_data.encode() if input_data else None
        stdout, stderr = await proc.communicate(input=stdin_bytes)
        return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")

    @staticmethod
    def _parse_kv(text: str, sep: str = ": ") -> dict[str, str]:
        """Parse key: value lines into a dict."""
        result: dict[str, str] = {}
        for line in text.strip().splitlines():
            if sep in line:
                k, v = line.split(sep, 1)
                result[k.strip()] = v.strip()
        return result

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def status(self) -> dict:
        """Return detailed repository status.

        Keys include ``branch``, ``ahead``, ``behind``, ``staged``,
        ``modified``, ``untracked``, ``conflicts``.
        """
        code, stdout, stderr = await self._run_git("status", "--porcelain=v2", "--branch")
        if code != 0:
            return {"error": stderr.strip(), "branch": "unknown"}

        branch = "unknown"
        ahead = 0
        behind = 0
        staged: list[str] = []
        modified: list[str] = []
        untracked: list[str] = []
        conflicts: list[str] = []

        for line in stdout.splitlines():
            if line.startswith("# branch.head "):
                branch = line.split(" ", 2)[-1]
            elif line.startswith("# branch.ab "):
                parts = line.split()
                ahead = int(parts[2].lstrip("+")) if len(parts) > 2 else 0
                behind = int(parts[3].lstrip("-")) if len(parts) > 3 else 0
            elif line.startswith("1 ") or line.startswith("2 "):
                parts = line.split()
                xy = parts[1] if len(parts) > 1 else ""
                path = parts[-1] if parts else ""
                if xy[0] in ("M", "A", "D", "R", "C"):
                    staged.append(path)
                if xy[1] in ("M", "D"):
                    modified.append(path)
                if xy[0] == "U" or xy[1] == "U":
                    conflicts.append(path)
            elif line.startswith("? "):
                untracked.append(line.split(" ", 1)[-1])

        return {
            "branch": branch,
            "ahead": ahead,
            "behind": behind,
            "staged": staged,
            "modified": modified,
            "untracked": untracked,
            "conflicts": conflicts,
        }

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    async def commit(
        self,
        message: Optional[str] = None,
        auto_stage: bool = True,
        amend: bool = False,
    ) -> dict:
        """Create a git commit.

        Args:
            message: Commit message.  When *None* one is generated automatically.
            auto_stage: Run ``git add -A`` before committing.
            amend: Amend the previous commit instead of creating a new one.

        Returns:
            Dict with keys ``sha``, ``message``, ``files``.
        """
        if auto_stage:
            await self._run_git("add", "-A")

        if message is None:
            message = await self.generate_commit_message()

        args = ["commit", "-m", message]
        if amend:
            args.append("--amend")

        code, stdout, stderr = await self._run_git(*args)

        if code != 0 and "nothing to commit" in stderr:
            return {"sha": None, "message": "", "files": 0, "status": "nothing_to_commit"}

        sha_match = re.search(r"\[.*?([0-9a-f]{7,40})\]", stderr) or re.search(
            r"([0-9a-f]{7,40})", stderr
        )
        sha = sha_match.group(1) if sha_match else None

        _, diff_out, _ = await self._run_git("diff", "--stat", "HEAD~1", "HEAD")

        return {
            "sha": sha,
            "message": message,
            "files": len(diff_out.strip().splitlines()) if diff_out.strip() else 0,
            "status": "ok" if code == 0 else "error",
        }

    # ------------------------------------------------------------------
    # Commit message generation
    # ------------------------------------------------------------------

    async def generate_commit_message(self) -> str:
        """Analyze staged changes and produce a conventional-commit message.

        Heuristics inspect filenames and diff content to pick the type
        (feat, fix, docs, refactor, …) and a short summary.
        """
        _, diff_out, _ = await self._run_git("diff", "--cached")

        if not diff_out.strip():
            # Nothing staged – look at working tree changes
            _, diff_out, _ = await self._run_git("diff")
            if not diff_out.strip():
                return "chore: update"

        files_changed: set[str] = set()
        additions = 0
        deletions = 0
        for line in diff_out.splitlines():
            m = re.match(r"\+\+\+ b/(.+)", line)
            if m:
                files_changed.add(m.group(1))
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
            if line.startswith("-") and not line.startswith("---"):
                deletions += 1

        commit_type = self._classify_commit_type(files_changed, diff_out)
        scope = self._extract_scope(files_changed)
        summary = self._summarize(files_changed, additions, deletions, diff_out)

        parts = [commit_type]
        if scope:
            parts.append(f"({scope})")
        parts.append(f": {summary}")
        return "".join(parts)

    @staticmethod
    def _classify_commit_type(files: set[str], diff: str) -> str:
        """Pick a conventional-commit type from file paths and diff content."""
        lower_diff = diff.lower()

        test_indicators = {"test", "spec", "__tests__", "_test", "_spec", "tests/", "conftest"}
        doc_indicators = {"readme", "license", "changelog", "docs/", "doc/", "copyright", "contributing"}
        ci_indicators = {".github/", ".gitlab-ci", ".circleci", "dockerfile", ".dockerignore",
                         "docker-compose", "makefile", "jenkins", ".travis"}

        has_test = any(test_ind in f.lower() for f in files)
        has_doc = any(doc_ind in f.lower() for f in files)
        has_ci = any(ci_ind in f.lower() for f in files)

        if has_test and not has_doc:
            return "test"
        if has_doc:
            return "docs"
        if has_ci:
            return "ci"

        # Inspect diff content
        if re.search(r"fix|bug|error|issue|broken|crash|regression", lower_diff):
            return "fix"
        if re.search(r"refactor|cleanup|reorganize|restructure", lower_diff):
            return "refactor"
        if re.search(r"perf|optim|speed|slow|faster|cache", lower_diff):
            return "perf"
        if re.search(r"feat|add|implement|support|introduce|create", lower_diff):
            return "feat"

        # Type by file extension
        exts = {os.path.splitext(f)[1].lower() for f in files}
        if exts & {".md", ".rst", ".txt"}:
            return "docs"
        if ".py" in exts or ".js" in exts or ".ts" in exts:
            return "feat"

        return "chore"

    @staticmethod
    def _extract_scope(files: set[str]) -> str:
        """Derive an optional scope from file paths."""
        if not files:
            return ""
        dirs = {f.split("/")[0] for f in files if "/" in f}
        if len(dirs) == 1:
            return dirs.pop()
        return ""

    @staticmethod
    def _summarize(files: set[str], additions: int, deletions: int, diff: str) -> str:
        """Build a short one-line summary of the changes."""
        names = [os.path.basename(f) for f in sorted(files)]
        if len(names) == 1:
            return f"update {names[0]}"
        if len(names) <= 3:
            return f"update {', '.join(names)}"
        if len(names) <= 6:
            return f"update {len(names)} files"
        return f"update {len(names)} files"

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    async def log(
        self,
        count: int = 20,
        author: Optional[str] = None,
        since: Optional[str] = None,
    ) -> list[dict]:
        """Return structured commit log entries."""
        args = [
            "log",
            f"-{count}",
            "--format=%H%n%h%n%an%n%ae%n%aI%n%s%n%b%n---END---",
        ]
        if author:
            args += ["--author", author]
        if since:
            args += ["--since", since]

        code, stdout, _ = await self._run_git(*args)
        if code != 0 or not stdout.strip():
            return []

        entries: list[dict] = []
        blocks = stdout.split("---END---")
        for block in blocks:
            lines = [l for l in block.strip().splitlines() if l.strip()]
            if len(lines) < 6:
                continue
            entries.append({
                "sha": lines[0],
                "short_sha": lines[1],
                "author": lines[2],
                "email": lines[3],
                "date": lines[4],
                "message": lines[5],
                "body": "\n".join(lines[6:]) if len(lines) > 6 else "",
            })
        return entries

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    async def diff(self, staged: bool = False, file: Optional[str] = None) -> str:
        """Return the diff as a string."""
        args = ["diff"]
        if staged:
            args.append("--cached")
        if file:
            args.extend(["--", file])
        code, stdout, _ = await self._run_git(*args)
        return stdout

    # ------------------------------------------------------------------
    # Branch
    # ------------------------------------------------------------------

    async def branch(self, name: str, checkout: bool = True) -> dict:
        """Create a new branch, optionally checking it out."""
        create_code, _, create_err = await self._run_git("branch", name)
        if create_code != 0:
            return {"status": "error", "message": create_err.strip()}

        result: dict[str, Any] = {"branch": name, "status": "created"}
        if checkout:
            co_code, _, co_err = await self._run_git("checkout", name)
            if co_code != 0:
                return {"status": "error", "message": co_err.strip()}
            result["status"] = "created_and_checked_out"

        return result

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    async def merge(self, branch: str, strategy: str = "merge") -> dict:
        """Merge *branch* using the given strategy.

        Supported strategies: ``merge``, ``rebase``, ``squash``.
        """
        if strategy == "rebase":
            code, stdout, stderr = await self._run_git("rebase", branch)
            return {
                "status": "ok" if code == 0 else "error",
                "strategy": "rebase",
                "branch": branch,
                "message": (stdout + stderr).strip(),
            }

        if strategy == "squash":
            code, stdout, stderr = await self._run_git("merge", "--squash", branch)
            if code != 0:
                return {"status": "error", "strategy": "squash", "branch": branch, "message": stderr.strip()}
            await self._run_git("commit", "-m", f"Squashed merge of '{branch}'")
            return {"status": "ok", "strategy": "squash", "branch": branch, "message": stdout.strip()}

        # default: merge
        code, stdout, stderr = await self._run_git("merge", branch)
        return {
            "status": "ok" if code == 0 else "error",
            "strategy": "merge",
            "branch": branch,
            "message": (stdout + stderr).strip(),
        }

    # ------------------------------------------------------------------
    # Cherry-pick
    # ------------------------------------------------------------------

    async def cherry_pick(self, commit_sha: str) -> dict:
        """Cherry-pick a commit onto the current branch."""
        code, stdout, stderr = await self._run_git("cherry-pick", commit_sha)
        return {
            "status": "ok" if code == 0 else "error",
            "sha": commit_sha,
            "message": (stdout + stderr).strip(),
        }

    # ------------------------------------------------------------------
    # Stash
    # ------------------------------------------------------------------

    async def stash(self, message: Optional[str] = None) -> dict:
        """Stash current changes."""
        args = ["stash", "push"]
        if message:
            args += ["-m", message]
        code, stdout, stderr = await self._run_git(*args)
        return {
            "status": "ok" if code == 0 else "error",
            "message": stderr.strip() or stdout.strip(),
        }

    async def stash_pop(self, index: int = 0) -> dict:
        """Pop a stash entry by index."""
        code, stdout, stderr = await self._run_git("stash", "pop", f"stash@{{{index}}}")
        return {
            "status": "ok" if code == 0 else "error",
            "index": index,
            "message": (stdout + stderr).strip(),
        }

    # ------------------------------------------------------------------
    # Tag
    # ------------------------------------------------------------------

    async def tag(self, name: str, message: Optional[str] = None) -> dict:
        """Create a git tag (annotated if *message* is provided)."""
        if message:
            code, _, stderr = await self._run_git("tag", "-a", name, "-m", message)
        else:
            code, _, stderr = await self._run_git("tag", name)
        return {
            "status": "ok" if code == 0 else "error",
            "tag": name,
            "message": stderr.strip(),
        }

    # ------------------------------------------------------------------
    # Blame
    # ------------------------------------------------------------------

    async def blame(self, file: str, line: Optional[int] = None) -> dict:
        """Return blame information for *file*, optionally for a single *line*."""
        args = ["blame", "--porcelain", file]
        code, stdout, stderr = await self._run_git(*args)
        if code != 0:
            return {"status": "error", "message": stderr.strip()}

        entries: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        for raw_line in stdout.splitlines():
            if re.match(r"^[0-9a-f]{40}", raw_line):
                if current.get("content") is not None:
                    entries.append(current)
                parts = raw_line.split()
                current = {
                    "sha": parts[0],
                    "original_line": int(parts[1]),
                    "final_line": int(parts[2]),
                    "content": None,
                }
            elif raw_line.startswith("author "):
                current["author"] = raw_line[7:]
            elif raw_line.startswith("author-mail "):
                current["email"] = raw_line[12:].strip("<>")
            elif raw_line.startswith("author-time "):
                ts = int(raw_line[12:])
                current["date"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            elif raw_line.startswith("\t"):
                current["content"] = raw_line[1:]

        if current.get("content") is not None:
            entries.append(current)

        if line is not None and 1 <= line <= len(entries):
            return {"status": "ok", "line": entries[line - 1]}

        return {"status": "ok", "lines": entries, "total": len(entries)}

    # ------------------------------------------------------------------
    # Undo (soft reset)
    # ------------------------------------------------------------------

    async def undo(self, n: int = 1) -> dict:
        """Soft-reset the last *n* commits, keeping changes staged."""
        if n < 1:
            return {"status": "error", "message": "n must be >= 1"}

        code, stdout, stderr = await self._run_git("reset", f"--soft", f"HEAD~{n}")
        return {
            "status": "ok" if code == 0 else "error",
            "commits_undone": n,
            "message": (stdout + stderr).strip(),
        }

    # ------------------------------------------------------------------
    # Contributors
    # ------------------------------------------------------------------

    async def get_contributors(self) -> list[dict]:
        """Return contributor statistics (name, email, commit count)."""
        code, stdout, _ = await self._run_git(
            "shortlog", "-sne", "--all",
        )
        if code != 0 or not stdout.strip():
            return []

        contributors: list[dict] = []
        for line in stdout.strip().splitlines():
            m = re.match(r"^\s*(\d+)\s+(.+)\s+<(.+)>$", line.strip())
            if m:
                contributors.append({
                    "commits": int(m.group(1)),
                    "name": m.group(2).strip(),
                    "email": m.group(3).strip(),
                })
        return contributors

    # ------------------------------------------------------------------
    # File history
    # ------------------------------------------------------------------

    async def get_file_history(self, filepath: str) -> list[dict]:
        """Return the commit history that touched *filepath*."""
        code, stdout, _ = await self._run_git(
            "log",
            "--format=%H%n%h%n%an%n%aI%n%s%n---END---",
            "--follow",
            "--", filepath,
        )
        if code != 0 or not stdout.strip():
            return []

        entries: list[dict] = []
        for block in stdout.split("---END---"):
            lines = [l for l in block.strip().splitlines() if l.strip()]
            if len(lines) >= 5:
                entries.append({
                    "sha": lines[0],
                    "short_sha": lines[1],
                    "author": lines[2],
                    "date": lines[3],
                    "message": lines[4],
                })
        return entries

    # ------------------------------------------------------------------
    # Search commits
    # ------------------------------------------------------------------

    async def search_commits(self, query: str) -> list[dict]:
        """Search commit messages for *query* (substring match)."""
        code, stdout, _ = await self._run_git(
            "log",
            "--all",
            "--format=%H%n%h%n%an%n%aI%n%s%n---END---",
            "--grep", query,
            "-i",
        )
        if code != 0 or not stdout.strip():
            return []

        entries: list[dict] = []
        for block in stdout.split("---END---"):
            lines = [l for l in block.strip().splitlines() if l.strip()]
            if len(lines) >= 5:
                entries.append({
                    "sha": lines[0],
                    "short_sha": lines[1],
                    "author": lines[2],
                    "date": lines[3],
                    "message": lines[4],
                })
        return entries

    # ------------------------------------------------------------------
    # Repo stats
    # ------------------------------------------------------------------

    async def get_repo_stats(self) -> dict:
        """Return aggregate repository statistics."""
        # Total commits
        _, commit_count_out, _ = await self._run_git("rev-list", "--count", "HEAD")
        total_commits = int(commit_count_out.strip()) if commit_count_out.strip() else 0

        # Lines added / removed (shortlog-style via log --stat)
        _, log_out, _ = await self._run_git(
            "log", "--all", "--shortstat", "--format=%H"
        )
        total_added = 0
        total_removed = 0
        for m in re.finditer(r"(\d+) insertion", log_out):
            total_added += int(m.group(1))
        for m in re.finditer(r"(\d+) deletion", log_out):
            total_removed += int(m.group(1))

        # Active branches
        _, branches_out, _ = await self._run_git("branch", "--list")
        active_branches = len([l for l in branches_out.strip().splitlines() if l.strip()])

        # Stash count
        _, stash_out, _ = await self._run_git("stash", "list")
        stash_count = len([l for l in stash_out.strip().splitlines() if l.strip()])

        # Current branch
        st = await self.status()

        return {
            "total_commits": total_commits,
            "lines_added": total_added,
            "lines_removed": total_removed,
            "active_branches": active_branches,
            "stash_count": stash_count,
            "current_branch": st.get("branch", "unknown"),
            "staged_files": len(st.get("staged", [])),
            "modified_files": len(st.get("modified", [])),
            "untracked_files": len(st.get("untracked", [])),
        }
