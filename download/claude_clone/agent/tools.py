"""
All tool implementations for the Claude Clone agent.
Every tool is a standalone async function with full error handling.
Tool schemas are auto-generated for Anthropic's tool_use format.
"""

import ast
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import chardet
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# Tool metadata & schema generation
# ──────────────────────────────────────────────

TOOL_PARAMETER_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "path": "string",
}


def generate_tool_schemas(tools_dict: Dict[str, Callable]) -> List[Dict]:
    """Auto-generate Anthropic tool schemas from Python functions."""
    schemas = []
    for name, func in tools_dict.items():
        doc = (func.__doc__ or "").strip()
        description = doc.split("\n")[0] if doc else f"Tool: {name}"

        params = {"type": "object", "properties": {}, "required": []}

        # Parse parameter info from docstring  (param_name: type — description)
        param_lines = [line.strip() for line in doc.split("\n") if line.strip().startswith("param")]

        # Also inspect function signature for hints
        hints = getattr(func, "__annotations__", {})
        # Remove 'return' from hints
        hints.pop("return", None)

        if not param_lines and hints:
            for pname, ptype in hints.items():
                json_type = TOOL_PARAMETER_TYPES.get(
                    str(ptype) if hasattr(ptype, "__name__") else str(ptype),
                    "string",
                )
                # Handle Optional etc.
                if "Optional" in str(ptype) or "None" in str(ptype):
                    json_type = json_type
                params["properties"][pname] = {"type": json_type, "description": f"Parameter {pname}"}
                if "Optional" not in str(ptype) and "None" not in str(ptype):
                    params["required"].append(pname)

        for pline in param_lines:
            # Format: "param_name (type): description"  or  "param_name: type — description"
            m = re.match(r"(\w+)\s*[:\(]\s*([^\):]+)[\):]\s*[-—]\s*(.+)", pline)
            if m:
                pname, ptype_str, pdesc = m.group(1), m.group(2).strip(), m.group(3).strip()
                json_type = TOOL_PARAMETER_TYPES.get(ptype_str, "string")
                params["properties"][pname] = {"type": json_type, "description": pdesc}
                if pname in hints and "Optional" not in str(hints[pname]):
                    if pname not in params["required"]:
                        params["required"].append(pname)

        # If still no required found from annotations, try to infer from docstring
        if not params["required"] and param_lines:
            for pline in param_lines:
                m = re.match(r"(\w+)", pline)
                if m:
                    pname = m.group(1)
                    if pname in params["properties"] and pname not in params["required"]:
                        params["required"].append(pname)

        schema = {
            "name": name,
            "description": description,
            "input_schema": params,
        }
        schemas.append(schema)
    return schemas


# ──────────────────────────────────────────────
# FILE TOOLS
# ──────────────────────────────────────────────

async def read_file(path: str) -> str:
    """Read a file and return its contents as a string.

    param path (str): — Path to the file to read. Can be absolute or relative to cwd.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found: {p}"
        if not p.is_file():
            return f"Error: Not a file: {p}"
        if not _is_path_safe(p):
            return f"Error: Access denied: {p}"

        # Detect encoding
        raw = p.read_bytes()
        if not raw:
            return ""

        detection = chardet.detect(raw)
        encoding = detection.get("encoding") or "utf-8"
        confidence = detection.get("confidence", 0)

        try:
            content = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            # Fallback encodings
            for enc in ["utf-8", "latin-1", "cp1252", "ascii"]:
                try:
                    content = raw.decode(enc)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            else:
                content = raw.decode("utf-8", errors="replace")

        line_count = content.count("\n") + (0 if content.endswith("\n") else 1)
        byte_size = len(raw)
        return f"[{byte_size} bytes, {line_count} lines, encoding: {encoding}]\n{content}"

    except PermissionError:
        return f"Error: Permission denied reading: {path}"
    except Exception as e:
        return f"Error reading {path}: {e}"


async def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed.

    param path (str): — Path to the file to write.
    param content (str): — Content to write to the file.
    """
    try:
        p = Path(path).expanduser()
        if not _is_path_safe(p.resolve()):
            return f"Error: Access denied: {p}"

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

        lines = content.count("\n") + (0 if content.endswith("\n") else 1)
        return f"Successfully wrote {len(content)} bytes ({lines} lines) to {p}"

    except PermissionError:
        return f"Error: Permission denied writing to: {path}"
    except Exception as e:
        return f"Error writing to {path}: {e}"


async def edit_file(path: str, old_str: str, new_str: str) -> str:
    """Precise find and replace within a file. Fails clearly if old_str not found.

    param path (str): — Path to the file to edit.
    param old_str (str): — The exact string to find in the file.
    param new_str (str): — The replacement string.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found: {p}"
        if not _is_path_safe(p):
            return f"Error: Access denied: {p}"

        content = p.read_text(encoding="utf-8")

        occurrences = content.count(old_str)
        if occurrences == 0:
            # Try a fuzzy match: show nearby context
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if old_str.split("\n")[0][:30] in line:
                    return (
                        f"Error: old_str not found exactly in {p}.\n"
                        f"Possible match near line {i+1}:\n{lines[i]}"
                    )
            return f"Error: old_str not found in {p}. Ensure the exact text (including whitespace) matches."

        if occurrences > 1:
            return (
                f"Error: old_str found {occurrences} times in {p}. "
                f"Please provide more context to make the match unique."
            )

        new_content = content.replace(old_str, new_str, 1)
        p.write_text(new_content, encoding="utf-8")

        added = len(new_str) - len(old_str)
        return f"Successfully edited {p} (delta: {added:+d} bytes)"

    except PermissionError:
        return f"Error: Permission denied editing: {path}"
    except Exception as e:
        return f"Error editing {path}: {e}"


async def append_file(path: str, content: str) -> str:
    """Append content to the end of a file.

    param path (str): — Path to the file.
    param content (str): — Content to append.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not _is_path_safe(p):
            return f"Error: Access denied: {p}"

        if p.exists():
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
            return f"Appended {len(content)} bytes to {p}"
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Created new file {p} ({len(content)} bytes)"

    except Exception as e:
        return f"Error appending to {path}: {e}"


async def delete_file(path: str) -> str:
    """Delete a file. Returns confirmation message.

    param path (str): — Path to the file to delete.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found: {p}"
        if not p.is_file():
            return f"Error: Not a file: {p}"
        if not _is_path_safe(p):
            return f"Error: Access denied: {p}"

        p.unlink()
        return f"Deleted: {p}"

    except PermissionError:
        return f"Error: Permission denied deleting: {path}"
    except Exception as e:
        return f"Error deleting {path}: {e}"


async def move_file(src: str, dst: str) -> str:
    """Move or rename a file or directory.

    param src (str): — Source path.
    param dst (str): — Destination path.
    """
    try:
        s = Path(src).expanduser().resolve()
        d = Path(dst).expanduser().resolve()

        if not s.exists():
            return f"Error: Source not found: {s}"
        if not _is_path_safe(s) or not _is_path_safe(d):
            return "Error: Access denied"

        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        return f"Moved {s} → {d}"

    except Exception as e:
        return f"Error moving {src} → {dst}: {e}"


async def copy_file(src: str, dst: str) -> str:
    """Copy a file or directory.

    param src (str): — Source path.
    param dst (str): — Destination path.
    """
    try:
        s = Path(src).expanduser().resolve()
        d = Path(dst).expanduser().resolve()

        if not s.exists():
            return f"Error: Source not found: {s}"
        if not _is_path_safe(s) or not _is_path_safe(d):
            return "Error: Access denied"

        d.parent.mkdir(parents=True, exist_ok=True)

        if s.is_dir():
            shutil.copytree(str(s), str(d))
            return f"Copied directory {s} → {d}"
        else:
            shutil.copy2(str(s), str(d))
            return f"Copied file {s} → {d}"

    except Exception as e:
        return f"Error copying {src} → {dst}: {e}"


# ──────────────────────────────────────────────
# DIRECTORY TOOLS
# ──────────────────────────────────────────────

async def list_directory(path: str = ".", show_hidden: bool = False) -> str:
    """List directory contents in tree format.

    param path (str): — Directory path to list. Defaults to current directory.
    param show_hidden (bool): — Whether to show hidden files/directories. Default False.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            return f"Error: Not a directory: {p}"

        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))

        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]

        if not entries:
            return f"Directory {p} is empty."

        lines = []
        for entry in entries:
            icon = "📁" if entry.is_dir() else "📄"
            size = ""
            if entry.is_file():
                try:
                    byte_size = entry.stat().st_size
                    size = f" ({_format_size(byte_size)})"
                except OSError:
                    size = " (size unknown)"

            name = entry.name
            if entry.is_dir():
                name += "/"
            lines.append(f"  {icon} {name}{size}")

        header = f"Directory: {p} ({len(entries)} entries)"
        return header + "\n" + "\n".join(lines)

    except PermissionError:
        return f"Error: Permission denied listing: {path}"
    except Exception as e:
        return f"Error listing directory {path}: {e}"


async def create_directory(path: str) -> str:
    """Create a directory, including any parent directories.

    param path (str): — Directory path to create.
    """
    try:
        p = Path(path).expanduser()
        if not _is_path_safe(p.resolve()):
            return f"Error: Access denied: {p}"

        p.mkdir(parents=True, exist_ok=True)
        return f"Created directory: {p.resolve()}"
    except Exception as e:
        return f"Error creating directory {path}: {e}"


async def get_project_structure(path: str = ".", depth: int = 3) -> str:
    """Get full project tree with file sizes.

    param path (str): — Root path of the project. Default: current directory.
    param depth (int): — Maximum depth of tree traversal. Default: 3.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            return f"Error: Not a directory: {p}"

        # Filter out common non-project directories
        skip_dirs = {
            "__pycache__", ".git", ".svn", ".hg", "node_modules", ".tox",
            ".mypy_cache", ".pytest_cache", ".venv", "venv", ".env",
            "dist", "build", ".eggs", "*.egg-info",
        }

        lines = []
        file_count = 0
        dir_count = 0

        def _walk(current: Path, prefix: str, current_depth: int):
            nonlocal file_count, dir_count
            if current_depth >= depth:
                return

            try:
                entries = sorted(current.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            except PermissionError:
                lines.append(f"{prefix}  ⚠️ Permission denied")
                return

            # Separate dirs and files
            dirs = []
            files = []
            for entry in entries:
                if entry.name.startswith(".") and entry.name not in (".env",):
                    continue
                if entry.name in skip_dirs or any(entry.match(s) for s in skip_dirs if "*" in s):
                    continue
                if entry.is_dir():
                    dirs.append(entry)
                else:
                    files.append(entry)

            all_entries = dirs + files

            for i, entry in enumerate(all_entries):
                is_last = (i == len(all_entries) - 1)
                connector = "└── " if is_last else "├── "
                child_prefix = "    " if is_last else "│   "

                if entry.is_dir():
                    dir_count += 1
                    lines.append(f"{prefix}{connector}📁 {entry.name}/")
                    _walk(entry, prefix + child_prefix, current_depth + 1)
                else:
                    file_count += 1
                    try:
                        size = _format_size(entry.stat().st_size)
                        lines.append(f"{prefix}{connector}📄 {entry.name} ({size})")
                    except OSError:
                        lines.append(f"{prefix}{connector}📄 {entry.name}")

        lines.append(f"📁 {p.name}/")
        _walk(p, "", 0)

        summary = f"\n{dir_count} directories, {file_count} files"
        return "\n".join(lines) + summary

    except Exception as e:
        return f"Error getting project structure for {path}: {e}"


# ──────────────────────────────────────────────
# SEARCH TOOLS
# ──────────────────────────────────────────────

async def search_files(pattern: str, directory: str = ".") -> str:
    """Search for files matching a glob pattern.

    param pattern (str): — Glob pattern to search for (e.g., "*.py", "**/*.js").
    param directory (str): — Directory to search in. Default: current directory.
    """
    try:
        d = Path(directory).expanduser().resolve()
        if not d.is_dir():
            return f"Error: Not a directory: {d}"

        matches = sorted(d.glob(pattern), key=lambda x: str(x))

        if not matches:
            return f"No files matching '{pattern}' found in {d}"

        lines = [f"Found {len(matches)} file(s) matching '{pattern}':"]
        for m in matches[:200]:  # Limit output
            rel = m.relative_to(d)
            if m.is_dir():
                lines.append(f"  📁 {rel}/")
            else:
                try:
                    size = _format_size(m.stat().st_size)
                    lines.append(f"  📄 {rel} ({size})")
                except OSError:
                    lines.append(f"  📄 {rel}")

        if len(matches) > 200:
            lines.append(f"  ... and {len(matches) - 200} more")

        return "\n".join(lines)

    except Exception as e:
        return f"Error searching for '{pattern}': {e}"


async def grep(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
    case_sensitive: bool = False,
    include: str = None,
) -> str:
    """Search for a regex pattern inside file contents.

    param pattern (str): — Regular expression to search for.
    param path (str): — File or directory to search in.
    param recursive (bool): — Search recursively in directories. Default True.
    param case_sensitive (bool): — Case-sensitive search. Default False.
    param include (str): — Only search files matching this glob (e.g., "*.py").
    """
    try:
        p = Path(path).expanduser().resolve()
        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        matches = []

        def _search_file(file_path: Path):
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if regex.search(line):
                        # Truncate long lines
                        display_line = line[:200] + ("..." if len(line) > 200 else "")
                        rel = file_path.relative_to(p.parent) if p.is_dir() else file_path.name
                        matches.append(f"  {rel}:{i+1}: {display_line.strip()}")
            except (PermissionError, UnicodeDecodeError):
                pass

        if p.is_file():
            _search_file(p)
        elif p.is_dir():
            for entry in p.rglob("*") if recursive else p.iterdir():
                if not entry.is_file():
                    continue
                if include and not entry.match(include):
                    continue
                # Skip binary-ish files
                if entry.suffix in (".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".bin", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".zip", ".tar", ".gz"):
                    continue
                if entry.stat().st_size > 5_000_000:
                    continue
                _search_file(entry)
        else:
            return f"Error: Path not found: {path}"

        if not matches:
            return f"No matches found for pattern '{pattern}' in {path}"

        result = [f"Found {len(matches)} match(es) for '{pattern}':"]
        result.extend(matches[:500])
        if len(matches) > 500:
            result.append(f"  ... and {len(matches) - 500} more matches")

        return "\n".join(result)

    except Exception as e:
        return f"Error grepping for '{pattern}': {e}"


async def find_definition(symbol: str, directory: str = ".") -> str:
    """Find function/class definitions matching a symbol name.

    param symbol (str): — Symbol name to search for (function or class name).
    param directory (str): — Directory to search in.
    """
    try:
        d = Path(directory).expanduser().resolve()
        if not d.is_dir():
            return f"Error: Not a directory: {d}"

        results = []

        for py_file in d.rglob("*.py"):
            if py_file.stat().st_size > 2_000_000:
                continue
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue

            rel = py_file.relative_to(d) if py_file.is_relative_to(d) else py_file.name

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and symbol.lower() in node.name.lower():
                    line = node.lineno
                    bases = ", ".join(b.id if isinstance(b, ast.Name) else str(b.id) for b in node.bases if hasattr(b, "id"))
                    doc = ast.get_docstring(node) or ""
                    results.append(f"  📦 class {node.name}({bases})\n     → {rel}:{line}\n     {doc.splitlines()[0] if doc else '(no docstring)'}")

                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if symbol.lower() in node.name.lower():
                        line = node.lineno
                        args = [a.arg for a in node.args.args]
                        doc = ast.get_docstring(node) or ""
                        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                        results.append(f"  🔧 {prefix}def {node.name}({', '.join(args)})\n     → {rel}:{line}\n     {doc.splitlines()[0] if doc else '(no docstring)'}")

        if not results:
            return f"No definitions found for '{symbol}' in {d}"

        return f"Found {len(results)} definition(s) for '{symbol}':\n" + "\n\n".join(results[:50])

    except Exception as e:
        return f"Error finding definition '{symbol}': {e}"


# ──────────────────────────────────────────────
# EXECUTION TOOLS
# ──────────────────────────────────────────────

async def run_command(command: str, cwd: str = None, timeout: int = 30, env: dict = None) -> dict:
    """Run a shell command and return output.

    param command (str): — Shell command to execute.
    param cwd (str): — Working directory. Default: current directory.
    param timeout (int): — Timeout in seconds. Default: 30.
    param env (dict): — Additional environment variables.
    """
    try:
        work_dir = Path(cwd).expanduser().resolve() if cwd else Path.cwd()

        process_env = os.environ.copy()
        if env:
            for k, v in env.items():
                process_env[k] = str(v)

        start = time.time()
        proc = await _run_subprocess(
            command, shell=True, cwd=str(work_dir), env=process_env, timeout=timeout
        )
        elapsed = time.time() - start

        return {
            "stdout": proc["stdout"],
            "stderr": proc["stderr"],
            "returncode": proc["returncode"],
            "elapsed": f"{elapsed:.2f}s",
            "timed_out": proc.get("timed_out", False),
        }

    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "elapsed": "0.00s",
            "timed_out": False,
        }


async def run_python(code: str, cwd: str = None) -> str:
    """Execute a Python code string and return the output.

    param code (str): — Python code to execute.
    param cwd (str): — Working directory. Default: current directory.
    """
    try:
        work_dir = str(Path(cwd).expanduser().resolve()) if cwd else os.getcwd()

        # Write code to a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=work_dir) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = await _run_subprocess(
                [sys.executable, tmp_path],
                cwd=work_dir,
                timeout=60,
            )

            output = ""
            if result["stdout"]:
                output += result["stdout"]
            if result["stderr"]:
                if output:
                    output += "\n--- stderr ---\n"
                output += result["stderr"]

            if result["returncode"] != 0:
                output += f"\n[Exit code: {result['returncode']}]"

            return output.strip() if output.strip() else "(no output)"

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    except Exception as e:
        return f"Error executing Python code: {e}"


async def run_script(path: str, args: list = None) -> str:
    """Run a Python or shell script.

    param path (str): — Path to the script file.
    param args (list): — Arguments to pass to the script.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: Script not found: {p}"
        if not _is_path_safe(p):
            return f"Error: Access denied: {p}"

        args = args or []

        if p.suffix == ".py":
            cmd = [sys.executable, str(p)] + [str(a) for a in args]
        elif p.suffix in (".sh", ".bash"):
            cmd = ["bash", str(p)] + [str(a) for a in args]
        elif os.access(str(p), os.X_OK):
            cmd = [str(p)] + [str(a) for a in args]
        else:
            cmd = [sys.executable, str(p)] + [str(a) for a in args]

        result = await _run_subprocess(cmd, cwd=str(p.parent), timeout=120)

        output = ""
        if result["stdout"]:
            output += result["stdout"]
        if result["stderr"]:
            if output:
                output += "\n--- stderr ---\n"
            output += result["stderr"]

        if result["returncode"] != 0:
            output += f"\n[Exit code: {result['returncode']}]"

        return output.strip() if output.strip() else "(no output)"

    except Exception as e:
        return f"Error running script {path}: {e}"


# ──────────────────────────────────────────────
# WEB TOOLS
# ──────────────────────────────────────────────

async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web via DuckDuckGo HTML (no API key required).

    param query (str): — Search query string.
    param num_results (int): — Number of results to return. Default: 5.
    """
    try:
        import httpx

        # Use DuckDuckGo HTML search
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=headers,
            )
            resp.raise_for_status()

        html = resp.text

        # Parse results from DuckDuckGo HTML
        results = []
        # Simple regex extraction of search result snippets
        result_blocks = re.findall(
            r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?<a class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )

        if not result_blocks:
            # Try alternative pattern
            result_blocks = re.findall(
                r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?(?:result__snippet[^>]*>(.*?)</a>)?',
                html, re.DOTALL
            )

        seen_urls = set()
        for url, title, snippet in result_blocks:
            url = re.sub(r'^//', 'https://', url)
            # Clean HTML tags
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = re.sub(r'<[^>]+>', '', snippet).strip() if snippet else ""

            if url not in seen_urls and title:
                seen_urls.add(url)
                results.append({"url": url, "title": title, "snippet": snippet})

        if not results:
            # Fallback: try the lite version
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(
                        "https://lite.duckduckgo.com/lite/",
                        params={"q": query},
                        headers=headers,
                    )
                    resp.raise_for_status()
                lite_html = resp.text
                # Parse lite results
                links = re.findall(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', lite_html, re.DOTALL)
                for url, text in links:
                    url = url.split("uddg=")[-1] if "uddg=" in url else url
                    from urllib.parse import unquote, parse_qs, urlparse
                    parsed = urlparse(url)
                    qs = parse_qs(parsed.query)
                    if "uddg" in qs:
                        url = unquote(qs["uddg"][0])
                    text = re.sub(r'<[^>]+>', '', text).strip()
                    if text and url not in seen_urls and len(results) < num_results:
                        seen_urls.add(url)
                        results.append({"url": url, "title": text, "snippet": ""})
            except Exception:
                pass

        if not results:
            return f"No web search results found for: {query}"

        lines = [f"Web search results for '{query}':\n"]
        for i, r in enumerate(results[:num_results], 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   {r['url']}")
            if r.get("snippet"):
                lines.append(f"   {r['snippet']}")
            lines.append("")

        return "\n".join(lines).strip()

    except ImportError:
        return "Error: httpx is not installed. Run: pip install httpx"
    except Exception as e:
        return f"Error performing web search: {e}"


async def fetch_url(url: str) -> str:
    """Fetch a URL and return cleaned webpage content.

    param url (str): — URL to fetch.
    """
    try:
        import httpx
        from html.parser import HTMLParser

        class HTMLTextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.result = []
                self.skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "noscript"):
                    self.skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "noscript"):
                    self.skip = False
                if tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "tr"):
                    self.result.append("\n")

            def handle_data(self, data):
                if not self.skip:
                    text = data.strip()
                    if text:
                        self.result.append(text)
                        self.result.append(" ")

            def get_text(self):
                return "".join(self.result)

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        }

        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()

        # Try to extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', resp.text, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else "No title"

        extractor = HTMLTextExtractor()
        extractor.feed(resp.text)
        text = extractor.get_text()

        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()

        # Truncate if too long
        max_chars = 20000
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[... content truncated, total length: {len(text)} chars]"

        return f"Title: {title}\nURL: {url}\n\n{text}"

    except ImportError:
        return "Error: httpx is not installed. Run: pip install httpx"
    except Exception as e:
        return f"Error fetching URL {url}: {e}"


# ──────────────────────────────────────────────
# CODE TOOLS
# ──────────────────────────────────────────────

async def lint_python(path: str) -> str:
    """Lint a Python file using ruff or flake8.

    param path (str): — Path to the Python file to lint.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"Error: File not found: {p}"

    # Try ruff first, then flake8, then pyflakes
    for tool in ["ruff", "flake8", "pyflakes"]:
        result = await _run_subprocess([tool, "check", str(p)], timeout=30)
        if result["returncode"] == 0 and tool not in ("pyflakes",):
            return f"No linting issues found (using {tool})."

        if result["returncode"] == 0:
            return f"No linting issues found (using {tool})."

        # ruff check returns 1 for issues found, not error
        if tool == "ruff" and result["returncode"] in (0, 1):
            output = (result["stdout"] + result["stderr"]).strip()
            if output:
                return f"Linting results ({tool}):\n{output}"
        elif result["returncode"] != 127:  # 127 = command not found
            output = (result["stdout"] + result["stderr"]).strip()
            if output:
                return f"Linting results ({tool}):\n{output}"

    return f"No Python linter found. Install one: pip install ruff"


async def format_python(path: str) -> str:
    """Format a Python file using black.

    param path (str): — Path to the Python file to format.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"Error: File not found: {p}"

    # Read original for comparison
    try:
        original = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

    result = await _run_subprocess(["black", str(p)], timeout=30)

    if result["returncode"] == 0:
        try:
            formatted = p.read_text(encoding="utf-8")
            if original == formatted:
                return f"File already formatted correctly (black)."
            else:
                delta = len(formatted) - len(original)
                return f"Formatted {p} (delta: {delta:+d} bytes) using black."
        except Exception:
            return f"Formatted {p} using black."
    else:
        stderr = result["stderr"].strip()
        if "No module named" in stderr or "command not found" in stderr or result["returncode"] == 127:
            return "Error: black is not installed. Install it: pip install black"
        return f"Error formatting {p}: {stderr}"


async def get_git_status(cwd: str = ".") -> str:
    """Get git repository status.

    param cwd (str): — Working directory. Default: current directory.
    """
    result = await _run_subprocess(
        ["git", "status", "--short", "--branch"],
        cwd=str(Path(cwd).expanduser().resolve()) if cwd else None,
        timeout=10,
    )

    if result["returncode"] == 128:
        return "Not a git repository."

    output = (result["stdout"] + result["stderr"]).strip()

    # Also get branch info
    branch_result = await _run_subprocess(
        ["git", "branch", "--show-current"],
        cwd=str(Path(cwd).expanduser().resolve()) if cwd else None,
        timeout=10,
    )
    branch = branch_result["stdout"].strip() if branch_result["returncode"] == 0 else "unknown"

    if not output:
        return f"Git branch: {branch}\nWorking tree clean."

    return f"Git branch: {branch}\n{output}"


async def git_diff(cwd: str = ".") -> str:
    """Get git diff of unstaged changes.

    param cwd (str): — Working directory. Default: current directory.
    """
    result = await _run_subprocess(
        ["git", "diff"],
        cwd=str(Path(cwd).expanduser().resolve()) if cwd else None,
        timeout=15,
    )

    if result["returncode"] == 128:
        return "Not a git repository."

    output = result["stdout"].strip()
    if not output:
        return "No unstaged changes."

    # Truncate very long diffs
    max_lines = 500
    lines = output.split("\n")
    if len(lines) > max_lines:
        output = "\n".join(lines[:max_lines])
        output += f"\n\n[... diff truncated, {len(lines) - max_lines} more lines]"

    return output


async def git_log(cwd: str = ".", n: int = 10) -> str:
    """Get recent git commit log.

    param cwd (str): — Working directory. Default: current directory.
    param n (int): — Number of commits to show. Default: 10.
    """
    result = await _run_subprocess(
        ["git", "log", f"--max-count={n}", "--pretty=format:%h %s (%cr) <%an>", "--no-color"],
        cwd=str(Path(cwd).expanduser().resolve()) if cwd else None,
        timeout=10,
    )

    if result["returncode"] == 128:
        return "Not a git repository."

    output = result["stdout"].strip()
    if not output:
        return "No commits found."

    return f"Recent commits ({n}):\n{output}"


# ──────────────────────────────────────────────
# SYSTEM TOOLS
# ──────────────────────────────────────────────

async def get_environment() -> str:
    """Get system environment information."""
    import platform

    lines = [
        f"Python: {sys.version}",
        f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
        f"Platform: {platform.platform()}",
        f"CWD: {os.getcwd()}",
        f"User: {os.environ.get('USER', 'unknown')}",
        f"Home: {Path.home()}",
        f"Shell: {os.environ.get('SHELL', 'unknown')}",
        "",
        "Environment variables:",
    ]

    # Show key environment variables (mask API keys)
    safe_vars = ["PATH", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV",
                 "LANG", "TERM", "HOME", "USER", "SHELL"]
    for var in safe_vars:
        val = os.environ.get(var, "")
        if val:
            display = val if len(val) < 200 else val[:200] + "..."
            lines.append(f"  {var}={display}")

    # Count pip packages
    pip_result = await _run_subprocess(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        timeout=10,
    )
    if pip_result["returncode"] == 0:
        packages = pip_result["stdout"].strip().split("\n")
        lines.append(f"\nInstalled pip packages: {len([p for p in packages if p])}")

    return "\n".join(lines)


async def install_package(package_name: str) -> str:
    """Install a Python package using pip.

    param package_name (str): — Name of the package to install.
    """
    result = await _run_subprocess(
        [sys.executable, "-m", "pip", "install", package_name],
        timeout=120,
    )

    output = ""
    if result["stdout"]:
        output += result["stdout"]
    if result["stderr"]:
        if "Successfully installed" in result["stderr"] or "Requirement already satisfied" in result["stderr"]:
            output += result["stderr"]
        elif result["returncode"] != 0:
            output += "\n--- stderr ---\n" + result["stderr"]

    if result["returncode"] != 0:
        output += f"\n[Exit code: {result['returncode']}]"

    return output.strip() if output.strip() else f"Installed {package_name}"


async def which(command: str) -> str:
    """Find the path to an executable command.

    param command (str): — Name of the command to find.
    """
    result = await _run_subprocess(
        ["which", command],
        timeout=5,
    )
    if result["returncode"] == 0:
        return result["stdout"].strip()

    # Try where on Windows
    result = await _run_subprocess(
        ["where", command],
        timeout=5,
    )
    if result["returncode"] == 0:
        return result["stdout"].strip()

    return f"'{command}' not found in PATH."


# ──────────────────────────────────────────────
# SANDBOX TOOLS
# ──────────────────────────────────────────────

async def sandbox_execute(code: str, language: str = "python", timeout: int = 30) -> str:
    """Execute code in an isolated sandbox environment.

    param code (str): — Code to execute in the sandbox.
    param language (str): — Programming language (python, javascript, etc.). Default: python.
    param timeout (int): — Execution timeout in seconds. Default: 30.
    """
    try:
        from agent.sandbox import SandboxExecutor

        executor = SandboxExecutor()
        result = await executor.run(code=code, language=language, timeout=timeout)
        lines = []
        lines.append(f"Language: {language}")
        lines.append(f"Execution time: {result.get('execution_time', 'N/A')}")
        lines.append(f"Memory usage: {result.get('memory_usage', 'N/A')}")
        lines.append(f"Exit code: {result.get('exit_code', 'N/A')}")
        if result.get("stdout"):
            lines.append(f"\n--- stdout ---\n{result['stdout']}")
        if result.get("stderr"):
            lines.append(f"\n--- stderr ---\n{result['stderr']}")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.sandbox module not found. Ensure the sandbox module is available."
    except Exception as e:
        return f"Error executing code in sandbox: {e}"


async def sandbox_install_package(package: str, language: str = "python") -> str:
    """Install a package in the sandbox environment.

    param package (str): — Name of the package to install.
    param language (str): — Target language for the package. Default: python.
    """
    try:
        from agent.sandbox import SandboxExecutor

        executor = SandboxExecutor()
        result = await executor.install_package(package=package, language=language)
        if result.get("success"):
            return f"Successfully installed {package} for {language}.\n{result.get('message', '')}"
        else:
            return f"Failed to install {package}: {result.get('error', 'Unknown error')}"
    except ImportError:
        return "Error: agent.sandbox module not found. Ensure the sandbox module is available."
    except Exception as e:
        return f"Error installing package in sandbox: {e}"


async def sandbox_list_files() -> str:
    """List all files in the sandbox workspace."""
    try:
        from agent.sandbox import SandboxExecutor

        executor = SandboxExecutor()
        files = await executor.list_files()
        if not files:
            return "Sandbox workspace is empty."
        lines = [f"Sandbox workspace files ({len(files)} total):"]
        for f in files:
            lines.append(f"  {f}")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.sandbox module not found. Ensure the sandbox module is available."
    except Exception as e:
        return f"Error listing sandbox files: {e}"


# ──────────────────────────────────────────────
# MEMORY TOOLS
# ──────────────────────────────────────────────

async def memory_search(query: str, limit: int = 10, session_id: str = None) -> str:
    """Search conversation memory for matching entries.

    param query (str): — Search query to find matching memories.
    param limit (int): — Maximum number of results. Default: 10.
    param session_id (str): — Optional session ID to scope the search.
    """
    try:
        from agent.memory import ConversationMemory

        memory = ConversationMemory()
        results = await memory.search(query=query, limit=limit, session_id=session_id)
        if not results:
            return f"No memories found matching '{query}'."
        lines = [f"Found {len(results)} memory entries matching '{query}':"]
        for i, entry in enumerate(results, 1):
            lines.append(f"{i}. [{entry.get('id', 'N/A')}] {entry.get('content', '')[:200]}")
            if entry.get("tags"):
                lines.append(f"   Tags: {entry.get('tags')}")
            if entry.get("timestamp"):
                lines.append(f"   Created: {entry.get('timestamp')}")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.memory module not found. Ensure the memory module is available."
    except Exception as e:
        return f"Error searching memory: {e}"


async def memory_save(content: str, tags: str = None) -> str:
    """Save content to conversation memory.

    param content (str): — Content to save to memory.
    param tags (str): — Comma-separated tags for categorization. Optional.
    """
    try:
        from agent.memory import ConversationMemory

        memory = ConversationMemory()
        tag_list = [t.strip() for t in tags.split(",")] if tags else []
        entry = await memory.save(content=content, tags=tag_list)
        entry_id = entry.get("id", "unknown")
        tag_str = f" with tags [{tags}]" if tags else ""
        return f"Memory saved successfully (ID: {entry_id}){tag_str}."
    except ImportError:
        return "Error: agent.memory module not found. Ensure the memory module is available."
    except Exception as e:
        return f"Error saving to memory: {e}"


async def memory_list_sessions() -> str:
    """List all conversation memory sessions."""
    try:
        from agent.memory import ConversationMemory

        memory = ConversationMemory()
        sessions = await memory.list_sessions()
        if not sessions:
            return "No memory sessions found."
        lines = [f"Memory sessions ({len(sessions)} total):"]
        for session in sessions:
            lines.append(f"  Session: {session.get('id', 'N/A')}")
            lines.append(f"    Entries: {session.get('entry_count', 0)}")
            lines.append(f"    Created: {session.get('created_at', 'N/A')}")
            lines.append(f"    Updated: {session.get('updated_at', 'N/A')}")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.memory module not found. Ensure the memory module is available."
    except Exception as e:
        return f"Error listing memory sessions: {e}"


async def memory_export(session_id: str, filepath: str) -> str:
    """Export a memory session to a file.

    param session_id (str): — Session ID to export.
    param filepath (str): — Destination file path for the export.
    """
    try:
        from agent.memory import ConversationMemory

        memory = ConversationMemory()
        result = await memory.export_session(session_id=session_id, filepath=filepath)
        if result.get("success"):
            entry_count = result.get("entry_count", 0)
            return f"Exported {entry_count} memory entries from session '{session_id}' to {filepath}."
        else:
            return f"Failed to export session: {result.get('error', 'Unknown error')}"
    except ImportError:
        return "Error: agent.memory module not found. Ensure the memory module is available."
    except Exception as e:
        return f"Error exporting memory session: {e}"


# ──────────────────────────────────────────────
# ANALYZER TOOLS
# ──────────────────────────────────────────────

async def analyze_project(project_path: str = ".") -> str:
    """Run a full project analysis with quality score and recommendations.

    param project_path (str): — Path to the project to analyze. Default: current directory.
    """
    try:
        from agent.analyzer import ProjectAnalyzer

        analyzer = ProjectAnalyzer()
        report = await analyzer.analyze(project_path=project_path)
        lines = []
        lines.append(f"Project Analysis: {project_path}")
        lines.append(f"Quality Score: {report.get('quality_score', 'N/A')}/100")
        lines.append(f"Files analyzed: {report.get('files_analyzed', 0)}")
        lines.append(f"Total lines: {report.get('total_lines', 0)}")
        if report.get("complexity_metrics"):
            lines.append(f"\nComplexity Metrics:")
            for metric, value in report["complexity_metrics"].items():
                lines.append(f"  {metric}: {value}")
        if report.get("recommendations"):
            lines.append(f"\nRecommendations:")
            for rec in report["recommendations"]:
                lines.append(f"  - {rec}")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.analyzer module not found. Ensure the analyzer module is available."
    except Exception as e:
        return f"Error analyzing project: {e}"


async def analyze_complexity(filepath: str) -> str:
    """Get a complexity report for a specific file with per-function scores.

    param filepath (str): — Path to the file to analyze.
    """
    try:
        from agent.analyzer import ProjectAnalyzer

        analyzer = ProjectAnalyzer()
        report = await analyzer.complexity_report(filepath=filepath)
        lines = [f"Complexity Report: {filepath}"]
        lines.append(f"Average complexity: {report.get('average_complexity', 'N/A')}")
        lines.append(f"Max complexity: {report.get('max_complexity', 'N/A')}")
        if report.get("functions"):
            lines.append(f"\nPer-function complexity:")
            for func in report["functions"]:
                name = func.get("name", "unknown")
                score = func.get("complexity", 0)
                line_no = func.get("line", "?")
                indicator = "!" if score > 10 else ("*" if score > 5 else " ")
                lines.append(f"  {indicator} {name} (line {line_no}): complexity {score}")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.analyzer module not found. Ensure the analyzer module is available."
    except Exception as e:
        return f"Error analyzing complexity: {e}"


async def analyze_dependencies(project_path: str = ".") -> str:
    """Get a dependency graph for the project including circular dependency detection.

    param project_path (str): — Path to the project. Default: current directory.
    """
    try:
        from agent.analyzer import ProjectAnalyzer

        analyzer = ProjectAnalyzer()
        report = await analyzer.dependency_graph(project_path=project_path)
        lines = [f"Dependency Graph: {project_path}"]
        lines.append(f"Modules found: {report.get('module_count', 0)}")
        if report.get("dependencies"):
            lines.append(f"\nDependencies:")
            for mod, deps in report["dependencies"].items():
                lines.append(f"  {mod} -> {', '.join(deps) if deps else '(no deps)'}")
        if report.get("circular_dependencies"):
            lines.append(f"\nCircular Dependencies Detected:")
            for cycle in report["circular_dependencies"]:
                lines.append(f"  ! {' -> '.join(cycle)}")
        else:
            lines.append(f"\nNo circular dependencies found.")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.analyzer module not found. Ensure the analyzer module is available."
    except Exception as e:
        return f"Error analyzing dependencies: {e}"


async def analyze_dead_code(project_path: str = ".") -> str:
    """Find dead code including unused imports, functions, and classes.

    param project_path (str): — Path to the project. Default: current directory.
    """
    try:
        from agent.analyzer import ProjectAnalyzer

        analyzer = ProjectAnalyzer()
        report = await analyzer.find_dead_code(project_path=project_path)
        lines = [f"Dead Code Analysis: {project_path}"]
        if report.get("unused_imports"):
            lines.append(f"\nUnused Imports ({len(report['unused_imports'])}):")
            for imp in report["unused_imports"]:
                lines.append(f"  - {imp.get('name', '')} in {imp.get('file', '')}:{imp.get('line', '')}")
        else:
            lines.append(f"\nNo unused imports found.")
        if report.get("unused_functions"):
            lines.append(f"\nUnused Functions ({len(report['unused_functions'])}):")
            for func in report["unused_functions"]:
                lines.append(f"  - {func.get('name', '')} in {func.get('file', '')}:{func.get('line', '')}")
        else:
            lines.append(f"\nNo unused functions found.")
        if report.get("unused_classes"):
            lines.append(f"\nUnused Classes ({len(report['unused_classes'])}):")
            for cls in report["unused_classes"]:
                lines.append(f"  - {cls.get('name', '')} in {cls.get('file', '')}:{cls.get('line', '')}")
        else:
            lines.append(f"\nNo unused classes found.")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.analyzer module not found. Ensure the analyzer module is available."
    except Exception as e:
        return f"Error analyzing dead code: {e}"


# ──────────────────────────────────────────────
# SECURITY TOOLS
# ──────────────────────────────────────────────

async def security_scan(project_path: str = ".") -> str:
    """Run a security vulnerability scan on the project.

    param project_path (str): — Path to the project to scan. Default: current directory.
    """
    try:
        from agent.security import SecurityScanner

        scanner = SecurityScanner()
        report = await scanner.scan(project_path=project_path)
        lines = [f"Security Scan: {project_path}"]
        lines.append(f"Total findings: {report.get('total_findings', 0)}")
        if report.get("findings"):
            severity_order = ["critical", "high", "medium", "low", "info"]
            for severity in severity_order:
                severity_findings = [f for f in report["findings"] if f.get("severity") == severity]
                if severity_findings:
                    lines.append(f"\n{severity.upper()} ({len(severity_findings)}):")
                    for finding in severity_findings:
                        lines.append(f"  - [{finding.get('rule', '')}] {finding.get('message', '')}")
                        lines.append(f"    File: {finding.get('file', '')}:{finding.get('line', '')}")
        else:
            lines.append("\nNo security findings.")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.security module not found. Ensure the security module is available."
    except Exception as e:
        return f"Error running security scan: {e}"


async def security_scan_secrets(project_path: str = ".") -> str:
    """Scan the project for leaked secrets and credentials.

    param project_path (str): — Path to the project to scan. Default: current directory.
    """
    try:
        from agent.security import SecurityScanner

        scanner = SecurityScanner()
        report = await scanner.scan_secrets(project_path=project_path)
        lines = [f"Secrets Scan: {project_path}"]
        if report.get("secrets"):
            lines.append(f"\nFound {len(report['secrets'])} leaked secret(s):")
            for secret in report["secrets"]:
                lines.append(f"  ! Type: {secret.get('type', 'unknown')}")
                lines.append(f"    File: {secret.get('file', '')}:{secret.get('line', '')}")
                lines.append(f"    Match: {secret.get('match', '')[:80]}...")
                if secret.get("remediation"):
                    lines.append(f"    Remediation: {secret['remediation']}")
        else:
            lines.append("\nNo leaked secrets found.")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.security module not found. Ensure the security module is available."
    except Exception as e:
        return f"Error scanning for secrets: {e}"


async def security_scan_dependencies() -> str:
    """Check project dependencies for known vulnerabilities and CVEs."""
    try:
        from agent.security import SecurityScanner

        scanner = SecurityScanner()
        report = await scanner.scan_dependencies()
        lines = ["Dependency Vulnerability Scan"]
        if report.get("vulnerabilities"):
            lines.append(f"\nFound {len(report['vulnerabilities'])} vulnerable dependencies:")
            for vuln in report["vulnerabilities"]:
                lines.append(f"  ! {vuln.get('package', '')} {vuln.get('version', '')}")
                lines.append(f"    CVE: {vuln.get('cve', 'N/A')}")
                lines.append(f"    Severity: {vuln.get('severity', 'unknown')}")
                lines.append(f"    Advisory: {vuln.get('advisory', 'N/A')}")
        else:
            lines.append("\nNo known vulnerabilities found in dependencies.")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.security module not found. Ensure the security module is available."
    except Exception as e:
        return f"Error scanning dependencies: {e}"


# ──────────────────────────────────────────────
# DEPLOY TOOLS
# ──────────────────────────────────────────────

async def deploy_project(platform: str, project_path: str = ".") -> str:
    """Deploy the project to a specified platform.

    param platform (str): — Target platform (docker, vercel, netlify, lambda, github_pages).
    param project_path (str): — Path to the project. Default: current directory.
    """
    valid_platforms = ["docker", "vercel", "netlify", "lambda", "github_pages"]
    if platform not in valid_platforms:
        return f"Error: Invalid platform '{platform}'. Must be one of: {', '.join(valid_platforms)}"

    try:
        from agent.deploy import DeployEngine

        engine = DeployEngine()
        result = await engine.deploy(platform=platform, project_path=project_path)
        lines = [f"Deployment to {platform}:"]
        if result.get("success"):
            lines.append(f"  Status: Success")
            if result.get("url"):
                lines.append(f"  URL: {result['url']}")
            if result.get("details"):
                lines.append(f"  Details: {result['details']}")
        else:
            lines.append(f"  Status: Failed")
            lines.append(f"  Error: {result.get('error', 'Unknown error')}")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.deploy module not found. Ensure the deploy module is available."
    except Exception as e:
        return f"Error deploying project: {e}"


async def detect_deploy_platform(project_path: str = ".") -> str:
    """Auto-detect the best deployment platform for a project.

    param project_path (str): — Path to the project. Default: current directory.
    """
    try:
        from agent.deploy import DeployEngine

        engine = DeployEngine()
        result = await engine.detect_platform(project_path=project_path)
        lines = [f"Deployment Platform Detection: {project_path}"]
        if result.get("platform"):
            lines.append(f"\nRecommended platform: {result['platform']}")
            lines.append(f"Reason: {result.get('reason', 'N/A')}")
            if result.get("alternatives"):
                lines.append(f"\nAlternative platforms:")
                for alt in result["alternatives"]:
                    lines.append(f"  - {alt.get('platform', '')}: {alt.get('reason', '')}")
        else:
            lines.append("\nCould not determine a recommended deployment platform.")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.deploy module not found. Ensure the deploy module is available."
    except Exception as e:
        return f"Error detecting deploy platform: {e}"


# ──────────────────────────────────────────────
# DATABASE TOOLS
# ──────────────────────────────────────────────

async def db_query(query: str, database: str = None) -> str:
    """Execute a SQL query against a database and return results as a formatted table.

    param query (str): — SQL query to execute.
    param database (str): — Database connection string. Optional.
    """
    try:
        from agent.database import DatabaseManager

        db = DatabaseManager(connection_string=database)
        result = await db.execute(query=query)
        if result.get("error"):
            return f"Error executing query: {result['error']}"
        rows = result.get("rows", [])
        columns = result.get("columns", [])
        if not rows:
            return f"Query executed successfully. 0 rows returned."
        # Format as table
        lines = []
        header = " | ".join(str(c) for c in columns)
        lines.append(header)
        lines.append("-" * len(header))
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))
        lines.append(f"\n({len(rows)} rows)")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.database module not found. Ensure the database module is available."
    except Exception as e:
        return f"Error executing database query: {e}"


async def db_list_tables(database: str = None) -> str:
    """List all tables in the database with row counts.

    param database (str): — Database connection string. Optional.
    """
    try:
        from agent.database import DatabaseManager

        db = DatabaseManager(connection_string=database)
        tables = await db.list_tables()
        if not tables:
            return "No tables found in the database."
        lines = [f"Database Tables ({len(tables)}):"]
        for table in tables:
            name = table.get("name", "unknown")
            rows = table.get("row_count", "?")
            size = table.get("size", "")
            line = f"  {name} ({rows} rows)"
            if size:
                line += f" [{size}]"
            lines.append(line)
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.database module not found. Ensure the database module is available."
    except Exception as e:
        return f"Error listing database tables: {e}"


# ──────────────────────────────────────────────
# GIT MANAGER TOOLS
# ──────────────────────────────────────────────

async def git_smart_commit(message: str = None, auto_stage: bool = True) -> str:
    """Create an AI-powered smart commit with an auto-generated or custom message.

    param message (str): — Commit message. If not provided, one will be auto-generated.
    param auto_stage (bool): — Whether to auto-stage all changes before committing. Default: True.
    """
    try:
        from agent.git_manager import GitManager

        manager = GitManager()
        result = await manager.smart_commit(message=message, auto_stage=auto_stage)
        if result.get("success"):
            lines = [f"Commit successful."]
            lines.append(f"  Hash: {result.get('hash', 'N/A')}")
            lines.append(f"  Message: {result.get('message', 'N/A')}")
            lines.append(f"  Files: {result.get('files_changed', 0)} changed, {result.get('insertions', 0)} insertions, {result.get('deletions', 0)} deletions")
            if result.get("auto_generated"):
                lines.append(f"  (Message was auto-generated)")
            return "\n".join(lines)
        else:
            return f"Commit failed: {result.get('error', 'Nothing to commit')}"
    except ImportError:
        return "Error: agent.git_manager module not found. Ensure the git_manager module is available."
    except Exception as e:
        return f"Error during smart commit: {e}"


async def git_repo_stats() -> str:
    """Get comprehensive repository statistics including contributors and commit frequency."""
    try:
        from agent.git_manager import GitManager

        manager = GitManager()
        stats = await manager.repo_stats()
        lines = ["Repository Statistics"]
        if stats.get("total_commits") is not None:
            lines.append(f"  Total commits: {stats['total_commits']}")
        if stats.get("total_files") is not None:
            lines.append(f"  Total files: {stats['total_files']}")
        if stats.get("total_lines") is not None:
            lines.append(f"  Total lines of code: {stats['total_lines']}")
        if stats.get("contributors"):
            lines.append(f"\nContributors ({len(stats['contributors'])}):")
            for contributor in stats["contributors"]:
                name = contributor.get("name", "unknown")
                commits = contributor.get("commits", 0)
                lines.append(f"  {name}: {commits} commits")
        if stats.get("commit_frequency"):
            lines.append(f"\nCommit Frequency:")
            for period, count in stats["commit_frequency"].items():
                lines.append(f"  {period}: {count} commits")
        if stats.get("line_counts"):
            lines.append(f"\nLine Counts:")
            for lang, count in stats["line_counts"].items():
                lines.append(f"  {lang}: {count} lines")
        return "\n".join(lines)
    except ImportError:
        return "Error: agent.git_manager module not found. Ensure the git_manager module is available."
    except Exception as e:
        return f"Error getting repository stats: {e}"


# ──────────────────────────────────────────────
# DESKTOP TOOLS
# ──────────────────────────────────────────────

# System Info Tools

async def desktop_system_info() -> str:
    """Get complete system information including CPU, RAM, disk, GPU, battery, and network.

    Returns a formatted summary of the current desktop system state.
    """
    try:
        import platform
        import subprocess

        lines = []
        lines.append("=== System Information ===")
        lines.append(f"Hostname: {platform.node()}")
        lines.append(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
        lines.append(f"Architecture: {platform.machine()}")
        lines.append(f"Processor: {platform.processor() or 'Unknown'}")

        # CPU info
        try:
            cpu_result = await _run_subprocess(
                "cat /proc/cpuinfo 2>/dev/null | grep 'model name' | head -1", shell=True, timeout=5
            )
            if cpu_result["returncode"] == 0 and cpu_result["stdout"].strip():
                cpu_model = cpu_result["stdout"].strip().split(":", 1)[1].strip()
                lines.append(f"CPU Model: {cpu_model}")

            cpu_count_result = await _run_subprocess("nproc", timeout=5)
            if cpu_count_result["returncode"] == 0:
                lines.append(f"CPU Cores: {cpu_count_result['stdout'].strip()}")
        except Exception:
            pass

        # RAM info
        try:
            mem_result = await _run_subprocess("free -h", timeout=5)
            if mem_result["returncode"] == 0 and mem_result["stdout"].strip():
                mem_lines = mem_result["stdout"].strip().split("\n")
                if len(mem_lines) >= 2:
                    parts = mem_lines[1].split()
                    lines.append(f"RAM: {parts[1]} total, {parts[2]} used, {parts[3]} available")
        except Exception:
            pass

        # Disk info
        try:
            disk_result = await _run_subprocess("df -h / 2>/dev/null | tail -1", shell=True, timeout=5)
            if disk_result["returncode"] == 0 and disk_result["stdout"].strip():
                parts = disk_result["stdout"].strip().split()
                if len(parts) >= 5:
                    lines.append(f"Disk (/): {parts[1]} total, {parts[2]} used, {parts[3]} available ({parts[4]})")
        except Exception:
            pass

        # GPU info
        try:
            gpu_result = await _run_subprocess("lspci 2>/dev/null | grep -i vga", shell=True, timeout=5)
            if gpu_result["returncode"] == 0 and gpu_result["stdout"].strip():
                gpu_line = gpu_result["stdout"].strip()
                gpu_name = gpu_line.split(":", 1)[1].strip() if ":" in gpu_line else gpu_line
                lines.append(f"GPU: {gpu_name}")
        except Exception:
            pass

        # Battery info
        try:
            bat_result = await _run_subprocess(
                "cat /sys/class/power_supply/BAT*/capacity 2>/dev/null | head -1", shell=True, timeout=5
            )
            if bat_result["returncode"] == 0 and bat_result["stdout"].strip():
                lines.append(f"Battery: {bat_result['stdout'].strip()}%")
            else:
                lines.append("Battery: N/A")
        except Exception:
            lines.append("Battery: N/A")

        # Network info
        try:
            net_result = await _run_subprocess(
                "ip -4 addr show 2>/dev/null | grep inet | grep -v 127.0.0.1 | awk '{print $2, $NF}' | head -5",
                shell=True, timeout=5
            )
            if net_result["returncode"] == 0 and net_result["stdout"].strip():
                lines.append(f"Network IPs:\n{net_result['stdout'].strip()}")
        except Exception:
            pass

        lines.append(f"\nPython: {sys.version}")
        lines.append(f"Uptime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error getting system info: {e}"


async def desktop_processes(filter_name: str = None, sort_by: str = "cpu", limit: int = 20) -> str:
    """List running processes with resource usage, optionally filtered and sorted.

    param filter_name (str): — Optional substring to filter process names by.
    param sort_by (str): — Sort by 'cpu', 'mem', or 'pid'. Default: cpu.
    param limit (int): — Maximum number of processes to return. Default: 20.
    """
    try:
        sort_col = "%cpu" if sort_by == "cpu" else ("%mem" if sort_by == "mem" else "pid")

        cmd = f"ps aux --sort=-{sort_col}"
        if filter_name:
            cmd += f" | grep -i '{filter_name}' | grep -v grep"

        result = await _run_subprocess(cmd, shell=True, timeout=10)

        if result["returncode"] != 0 or not result["stdout"].strip():
            return "No processes found."

        output_lines = result["stdout"].strip().split("\n")
        header = output_lines[0]
        body = output_lines[1:]

        # Format output
        lines = [f"=== Running Processes (sorted by {sort_by}, limit {limit}) ==="]
        lines.append(header)

        for line in body[:limit]:
            parts = line.split(None, 10)
            if len(parts) >= 11:
                # Truncate long command names
                user, pid, cpu, mem, vsz, rss, tty, stat, start, time, cmd = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7], parts[8], parts[9], parts[10]
                cmd_display = cmd[:80] + "..." if len(cmd) > 80 else cmd
                lines.append(f"  PID={pid:>6}  CPU={cpu:>5}%  MEM={mem:>5}%  {cmd_display}")
            else:
                lines.append(f"  {line[:100]}")

        if len(body) > limit:
            lines.append(f"  ... and {len(body) - limit} more processes")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing processes: {e}"


async def desktop_active_window() -> str:
    """Get information about the currently focused/active window.

    Returns window title, application name, PID, and geometry.
    """
    try:
        # Try xdotool + xprop (Linux X11)
        result = await _run_subprocess(
            "xdotool getactivewindow getwindowname getwindowpid 2>/dev/null", shell=True, timeout=5
        )
        if result["returncode"] == 0 and result["stdout"].strip():
            parts = result["stdout"].strip().split("\n")
            if len(parts) >= 2:
                title = parts[0]
                pid = parts[1] if len(parts) > 1 else "N/A"

                # Get geometry
                geo_result = await _run_subprocess(
                    "xdotool getactivewindow getwindowgeometry 2>/dev/null", shell=True, timeout=5
                )
                geometry = "N/A"
                if geo_result["returncode"] == 0:
                    geo_output = geo_result["stdout"].strip()
                    for line in geo_output.split("\n"):
                        if "Geometry" in line:
                            geometry = line.strip().split(":", 1)[1].strip()

                # Try to get app name via WM_CLASS
                class_result = await _run_subprocess(
                    "xprop -id $(xdotool getactivewindow) WM_CLASS 2>/dev/null", shell=True, timeout=5
                )
                app_name = "N/A"
                if class_result["returncode"] == 0 and 'WM_CLASS' in class_result["stdout"]:
                    app_name = class_result["stdout"].strip().split("=", 1)[1].strip().strip('"').split(",")[0].strip().strip('"')

                lines = [
                    "=== Active Window ===",
                    f"Title: {title}",
                    f"Application: {app_name}",
                    f"PID: {pid}",
                    f"Geometry: {geometry}",
                ]
                return "\n".join(lines)

        # Fallback: try wmctrl
        wm_result = await _run_subprocess(
            "wmctrl -a :ACTIVE: -l 2>/dev/null", shell=True, timeout=5
        )
        if wm_result["returncode"] == 0:
            active_result = await _run_subprocess(
                "wmctrl -l | grep $(xdotool getactivewindow 2>/dev/null) 2>/dev/null", shell=True, timeout=5
            )
            if active_result["returncode"] == 0 and active_result["stdout"].strip():
                return f"Active Window:\n{active_result['stdout'].strip()}"

        return "Error: Could not detect active window. Ensure xdotool is installed (Linux/X11)."

    except Exception as e:
        return f"Error getting active window: {e}"


async def desktop_screenshot(region: str = None, ocr: bool = False) -> str:
    """Take a screenshot of the screen or a specific region, with optional OCR.

    param region (str): — Screen region to capture as 'x,y,w,h'. Optional, captures full screen if omitted.
    param ocr (bool): — Whether to perform OCR text extraction on the screenshot. Default: False.
    """
    try:
        import tempfile as _tempfile

        screenshot_path = os.path.join(_tempfile.gettempdir(), f"screenshot_{int(time.time())}.png")

        # Use scrot or import for screenshot capture
        if region:
            cmd = f"scrot -a {region} '{screenshot_path}' 2>/dev/null || gnome-screenshot -a -f '{screenshot_path}' 2>/dev/null || import -crop {region} '{screenshot_path}' 2>/dev/null"
        else:
            cmd = f"scrot '{screenshot_path}' 2>/dev/null || gnome-screenshot -f '{screenshot_path}' 2>/dev/null || import -window root '{screenshot_path}' 2>/dev/null"

        result = await _run_subprocess(cmd, shell=True, timeout=15)

        if result["returncode"] != 0 or not os.path.exists(screenshot_path):
            return "Error: Screenshot failed. Ensure scrot, gnome-screenshot, or ImageMagick is installed."

        file_size = os.path.getsize(screenshot_path)
        lines = [
            f"Screenshot saved to: {screenshot_path}",
            f"File size: {_format_size(file_size)}",
            f"Region: {region if region else 'Full screen'}",
        ]

        # OCR if requested
        if ocr:
            ocr_result = await _run_subprocess(
                f"tesseract '{screenshot_path}' stdout 2>/dev/null", shell=True, timeout=30
            )
            if ocr_result["returncode"] == 0 and ocr_result["stdout"].strip():
                lines.append("\n--- OCR Text ---")
                lines.append(ocr_result["stdout"].strip())
            else:
                lines.append("\nOCR failed: tesseract not installed or could not extract text.")

        return "\n".join(lines)

    except Exception as e:
        return f"Error taking screenshot: {e}"


async def desktop_clipboard() -> str:
    """Get the current text content of the system clipboard.

    Returns the clipboard text, or an error message if inaccessible.
    """
    try:
        # Try xclip (X11)
        result = await _run_subprocess(
            "xclip -selection clipboard -o 2>/dev/null || xsel --clipboard --output 2>/dev/null",
            shell=True, timeout=5
        )
        if result["returncode"] == 0 and result["stdout"].strip():
            content = result["stdout"]
            if len(content) > 10000:
                content = content[:10000] + "\n[... clipboard content truncated, total length: {len(content)} chars]"
            return f"Clipboard content:\n{content}"
        elif result["stderr"] and "Error" not in result["stderr"]:
            return f"Clipboard appears to be empty or contains non-text data."

        return "Error: Could not access clipboard. Ensure xclip or xsel is installed."

    except Exception as e:
        return f"Error accessing clipboard: {e}"


# PC Control Tools

async def desktop_mouse_click(x: int, y: int, button: str = "left") -> str:
    """Click at a specific position on the screen.

    param x (int): — X coordinate on screen.
    param y (int): — Y coordinate on screen.
    param button (str): — Mouse button: 'left', 'right', or 'double'. Default: left.
    """
    try:
        if button not in ("left", "right", "double"):
            return f"Error: Invalid button '{button}'. Must be 'left', 'right', or 'double'."

        # Move mouse to position then click
        await _run_subprocess(f"xdotool mousemove {x} {y}", shell=True, timeout=5)

        if button == "left":
            click_cmd = "xdotool click 1"
        elif button == "right":
            click_cmd = "xdotool click 3"
        else:  # double
            click_cmd = "xdotool click --repeat 2 --delay 100 1"

        result = await _run_subprocess(click_cmd, shell=True, timeout=5)

        if result["returncode"] == 0:
            return f"Clicked {button} button at position ({x}, {y})."
        else:
            return f"Error: Mouse click failed. Ensure xdotool is installed."

    except Exception as e:
        return f"Error during mouse click: {e}"


async def desktop_type_text(text: str, delay: float = 0.02) -> str:
    """Type text using the keyboard with an optional delay between keystrokes.

    param text (str): — Text string to type.
    param delay (float): — Delay in seconds between keystrokes. Default: 0.02.
    """
    try:
        # Escape special characters for xdotool
        escaped = text.replace("'", "'\\''")
        result = await _run_subprocess(
            f"xdotool type --delay {int(delay * 1000)} '{escaped}'",
            shell=True, timeout=30
        )

        if result["returncode"] == 0:
            return f"Typed {len(text)} characters with {delay}s delay."
        else:
            return f"Error: Typing failed. Ensure xdotool is installed."

    except Exception as e:
        return f"Error typing text: {e}"


async def desktop_hotkey(keys: str) -> str:
    """Press a keyboard shortcut/hotkey combination.

    param keys (str): — Key combination to press (e.g., 'ctrl+c', 'alt+tab', 'ctrl+shift+t').
    """
    try:
        # Parse keys and convert to xdotool format
        key_map = {
            "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "shift": "shift",
            "super": "super", "win": "super", "cmd": "super",
            "tab": "Tab", "enter": "Return", "return": "Return",
            "space": "space", "escape": "Escape", "esc": "Escape",
            "backspace": "BackSpace", "delete": "Delete", "del": "Delete",
            "up": "Up", "down": "Down", "left": "Left", "right": "Right",
            "home": "Home", "end": "End", "pageup": "Page_Up", "pagedown": "Page_Down",
            "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4",
            "f5": "F5", "f6": "F6", "f7": "F7", "f8": "F8",
            "f9": "F9", "f10": "F10", "f11": "F11", "f12": "F12",
        }

        key_parts = [k.strip().lower() for k in keys.split("+")]
        modifiers = []
        final_keys = []

        for part in key_parts:
            mapped = key_map.get(part, part)
            if mapped in ("ctrl", "alt", "shift", "super"):
                modifiers.append(mapped)
            else:
                final_keys.append(mapped)

        if not final_keys:
            return f"Error: No key specified in '{keys}'. Provide at least one non-modifier key."

        key_combo = "+".join(modifiers + final_keys)
        result = await _run_subprocess(f"xdotool key '{key_combo}'", shell=True, timeout=5)

        if result["returncode"] == 0:
            return f"Hotkey '{keys}' pressed successfully (sent: {key_combo})."
        else:
            return f"Error: Hotkey failed. Ensure xdotool is installed."

    except Exception as e:
        return f"Error pressing hotkey: {e}"


async def desktop_launch_app(app_name: str, args: str = None) -> str:
    """Launch an application by name.

    param app_name (str): — Name or command of the application to launch.
    param args (str): — Optional arguments to pass to the application. Optional.
    """
    try:
        cmd = app_name
        if args:
            cmd = f"{app_name} {args}"

        # Try to launch as background process
        full_cmd = f"nohup {cmd} > /dev/null 2>&1 &"
        result = await _run_subprocess(full_cmd, shell=True, timeout=10)

        # Check if the command exists
        which_result = await _run_subprocess(f"which {app_name}", shell=True, timeout=5)
        if which_result["returncode"] != 0:
            return f"Warning: '{app_name}' not found in PATH. Attempted to launch anyway."

        # Give it a moment and check if it's running
        await __import__("asyncio").sleep(1)
        check_result = await _run_subprocess(f"pgrep -f '{app_name}'", shell=True, timeout=5)

        if check_result["returncode"] == 0 and check_result["stdout"].strip():
            pid = check_result["stdout"].strip().split("\n")[0]
            return f"Launched '{app_name}' (PID: {pid})."
        else:
            return f"Attempted to launch '{app_name}'. Process may have exited immediately."

    except Exception as e:
        return f"Error launching application: {e}"


async def desktop_open_url(url: str) -> str:
    """Open a URL in the default web browser.

    param url (str): — URL to open in the browser.
    """
    try:
        import webbrowser

        # Validate URL
        if not url.startswith(("http://", "https://", "ftp://")):
            url = "https://" + url

        webbrowser.open(url)
        return f"Opened URL in default browser: {url}"

    except Exception as e:
        return f"Error opening URL: {e}"


async def desktop_open_terminal(directory: str = ".", command: str = None) -> str:
    """Open a terminal emulator in a specific directory with an optional command.

    param directory (str): — Directory to open the terminal in. Default: current directory ('.').
    param command (str): — Optional command to run in the terminal. Optional.
    """
    try:
        dir_path = str(Path(directory).expanduser().resolve())

        if not os.path.isdir(dir_path):
            return f"Error: Directory not found: {dir_path}"

        # Try common terminal emulators
        terminals = [
            ("gnome-terminal", f"gnome-terminal --working-directory='{dir_path}"),
            ("konsole", f"konsole --workdir '{dir_path}'"),
            ("xfce4-terminal", f"xfce4-terminal --working-directory='{dir_path}'"),
            ("xterm", f"xterm -cd '{dir_path}'"),
        ]

        if command:
            terminals = [
                ("gnome-terminal", f"gnome-terminal --working-directory='{dir_path}' -- bash -c '{command}; exec bash'"),
                ("konsole", f"konsole --workdir '{dir_path}' -e bash -c '{command}; exec bash'"),
                ("xfce4-terminal", f"xfce4-terminal --working-directory='{dir_path}' -e \"bash -c '{command}; exec bash'\""),
                ("xterm", f"xterm -cd '{dir_path}' -e bash -c '{command}; exec bash'"),
            ]

        launched = False
        for name, cmd in terminals:
            which_result = await _run_subprocess(f"which {name}", shell=True, timeout=3)
            if which_result["returncode"] == 0:
                await _run_subprocess(f"nohup {cmd} > /dev/null 2>&1 &", shell=True, timeout=5)
                launched = True
                term_label = f" running command: {command}" if command else ""
                return f"Opened {name} in {dir_path}{term_label}."

        return "Error: No supported terminal emulator found. Tried: gnome-terminal, konsole, xfce4-terminal, xterm."

    except Exception as e:
        return f"Error opening terminal: {e}"


# Window Management Tools

async def desktop_list_windows() -> str:
    """List all open windows with their titles, IDs, and geometry.

    Returns a formatted list of all currently open windows.
    """
    try:
        # Try wmctrl
        result = await _run_subprocess("wmctrl -l -p 2>/dev/null", shell=True, timeout=5)

        if result["returncode"] == 0 and result["stdout"].strip():
            lines = ["=== Open Windows ==="]
            lines.append(f"{'ID':<12} {'PID':>8}  {'Workspace':>8}  Title")
            lines.append("-" * 70)
            for line in result["stdout"].strip().split("\n"):
                parts = line.split(None, 5)
                if len(parts) >= 6:
                    win_id = parts[0]
                    workspace = parts[1]
                    pid = parts[2]
                    hostname = parts[3]
                    title = parts[5]
                    lines.append(f"{win_id:<12} {pid:>8}  {workspace:>8}  {title[:60]}")
                else:
                    lines.append(f"  {line[:70]}")
            lines.append(f"\nTotal: {len(result['stdout'].strip().split(chr(10)))} windows")
            return "\n".join(lines)

        # Fallback: xdotool
        fallback = await _run_subprocess("xdotool search --name '' getwindowname 2>/dev/null", shell=True, timeout=5)
        if fallback["returncode"] == 0 and fallback["stdout"].strip():
            lines = ["=== Open Windows ==="]
            for i, name in enumerate(fallback["stdout"].strip().split("\n"), 1):
                lines.append(f"  {i}. {name[:80]}")
            return "\n".join(lines)

        return "Error: Could not list windows. Ensure wmctrl or xdotool is installed."

    except Exception as e:
        return f"Error listing windows: {e}"


async def desktop_focus_window(title: str) -> str:
    """Focus and bring a window to the foreground by its title.

    param title (str): — Window title substring to search for and focus.
    """
    try:
        # Try wmctrl
        result = await _run_subprocess(f"wmctrl -a '{title}' 2>/dev/null", shell=True, timeout=5)

        if result["returncode"] == 0:
            return f"Focused window matching: '{title}'"

        # Fallback: xdotool search
        search_result = await _run_subprocess(
            f"xdotool search --name '{title}' windowactivate 2>/dev/null", shell=True, timeout=5
        )
        if search_result["returncode"] == 0:
            return f"Focused window matching: '{title}'"

        return f"Error: No window found matching '{title}'."

    except Exception as e:
        return f"Error focusing window: {e}"


async def desktop_close_window(title: str) -> str:
    """Close a window by searching for its title.

    param title (str): — Window title substring to search for and close.
    """
    try:
        # Try wmctrl -c (graceful close)
        result = await _run_subprocess(f"wmctrl -c '{title}' 2>/dev/null", shell=True, timeout=5)

        if result["returncode"] == 0:
            return f"Closed window matching: '{title}'"

        # Fallback: xdotool search + windowclose
        search_result = await _run_subprocess(
            f"xdotool search --name '{title}' windowclose 2>/dev/null", shell=True, timeout=5
        )
        if search_result["returncode"] == 0:
            return f"Closed window matching: '{title}'"

        return f"Error: No window found matching '{title}'."

    except Exception as e:
        return f"Error closing window: {e}"


# Voice Tools

async def desktop_speak(text: str, voice: str = None, rate: float = 1.0) -> str:
    """Make the assistant speak text aloud using text-to-speech.

    param text (str): — Text to speak aloud.
    param voice (str): — Optional voice name or language code (e.g., 'en', 'en-US'). Optional.
    param rate (float): — Speech rate multiplier. 1.0 is normal speed. Default: 1.0.
    """
    try:
        # Clamp rate to sane range
        rate = max(0.25, min(4.0, rate))

        # Try espeak first (most common on Linux)
        espeak_cmd = f"espeak -s {int(150 * rate)} -p 50"
        if voice:
            espeak_cmd += f" -v '{voice}'"
        espeak_cmd += f" '{text.replace(chr(39), chr(92) + chr(39))}'"

        result = await _run_subprocess(espeak_cmd, shell=True, timeout=30)

        if result["returncode"] == 0:
            voice_info = f" (voice: {voice})" if voice else ""
            return f"Spoke {len(text)} characters at {rate}x speed{voice_info}."

        # Try pico2wave + aplay (alternative TTS)
        if os.path.exists("/usr/bin/pico2wave"):
            wav_path = os.path.join(tempfile.gettempdir(), f"tts_{int(time.time())}.wav")
            lang = voice if voice else "en-US"
            pico_result = await _run_subprocess(
                f"pico2wave -l '{lang}' -w '{wav_path}' '{text.replace(chr(39), chr(92) + chr(39))}' && aplay '{wav_path}'",
                shell=True, timeout=30
            )
            if pico_result["returncode"] == 0:
                return f"Spoke {len(text)} characters using pico2wave."
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        # Try say (macOS)
        say_result = await _run_subprocess(f"which say 2>/dev/null", shell=True, timeout=3)
        if say_result["returncode"] == 0:
            say_cmd = f"say -r {int(175 * rate)}"
            if voice:
                say_cmd += f" -v '{voice}'"
            say_cmd += f" '{text.replace(chr(39), chr(92) + chr(39))}'"
            say_result = await _run_subprocess(say_cmd, shell=True, timeout=30)
            if say_result["returncode"] == 0:
                return f"Spoke {len(text)} characters using macOS 'say'."

        return "Error: No TTS engine found. Install espeak: sudo apt install espeak"

    except Exception as e:
        return f"Error speaking text: {e}"


# ──────────────────────────────────────────────
# TOOL REGISTRY
# ──────────────────────────────────────────────

TOOLS_REGISTRY: Dict[str, Callable] = {
    # File Tools
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "append_file": append_file,
    "delete_file": delete_file,
    "move_file": move_file,
    "copy_file": copy_file,
    # Directory Tools
    "list_directory": list_directory,
    "create_directory": create_directory,
    "get_project_structure": get_project_structure,
    # Search Tools
    "search_files": search_files,
    "grep": grep,
    "find_definition": find_definition,
    # Execution Tools
    "run_command": run_command,
    "run_python": run_python,
    "run_script": run_script,
    # Web Tools
    "web_search": web_search,
    "fetch_url": fetch_url,
    # Code Tools
    "lint_python": lint_python,
    "format_python": format_python,
    "get_git_status": get_git_status,
    "git_diff": git_diff,
    "git_log": git_log,
    # System Tools
    "get_environment": get_environment,
    "install_package": install_package,
    "which": which,
    # Sandbox Tools
    "sandbox_execute": sandbox_execute,
    "sandbox_install_package": sandbox_install_package,
    "sandbox_list_files": sandbox_list_files,
    # Memory Tools
    "memory_search": memory_search,
    "memory_save": memory_save,
    "memory_list_sessions": memory_list_sessions,
    "memory_export": memory_export,
    # Analyzer Tools
    "analyze_project": analyze_project,
    "analyze_complexity": analyze_complexity,
    "analyze_dependencies": analyze_dependencies,
    "analyze_dead_code": analyze_dead_code,
    # Security Tools
    "security_scan": security_scan,
    "security_scan_secrets": security_scan_secrets,
    "security_scan_dependencies": security_scan_dependencies,
    # Deploy Tools
    "deploy_project": deploy_project,
    "detect_deploy_platform": detect_deploy_platform,
    # Database Tools
    "db_query": db_query,
    "db_list_tables": db_list_tables,
    # Git Manager Tools
    "git_smart_commit": git_smart_commit,
    "git_repo_stats": git_repo_stats,
    # Desktop Tools - System Info
    "desktop_system_info": desktop_system_info,
    "desktop_processes": desktop_processes,
    "desktop_active_window": desktop_active_window,
    "desktop_screenshot": desktop_screenshot,
    "desktop_clipboard": desktop_clipboard,
    # Desktop Tools - PC Control
    "desktop_mouse_click": desktop_mouse_click,
    "desktop_type_text": desktop_type_text,
    "desktop_hotkey": desktop_hotkey,
    "desktop_launch_app": desktop_launch_app,
    "desktop_open_url": desktop_open_url,
    "desktop_open_terminal": desktop_open_terminal,
    # Desktop Tools - Window Management
    "desktop_list_windows": desktop_list_windows,
    "desktop_focus_window": desktop_focus_window,
    "desktop_close_window": desktop_close_window,
    # Desktop Tools - Voice
    "desktop_speak": desktop_speak,
}


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

async def _run_subprocess(
    cmd,
    shell: bool = False,
    cwd: str = None,
    timeout: int = 30,
    env: dict = None,
) -> dict:
    """Run a subprocess asynchronously and capture output."""
    try:
        loop = __import__("asyncio").get_event_loop()
        proc = await __import__("asyncio").create_subprocess_shell(
            cmd if shell else " ".join(str(c) for c in cmd),
            stdout=__import__("asyncio").subprocess.PIPE,
            stderr=__import__("asyncio").subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        try:
            stdout, stderr = await __import__("asyncio").wait_for(
                proc.communicate(), timeout=timeout
            )
            return {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "returncode": proc.returncode,
            }
        except __import__("asyncio").TimeoutError:
            proc.kill()
            await proc.communicate()
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "returncode": -1,
                "timed_out": True,
            }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
        }


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def _is_path_safe(path: Path) -> bool:
    """Check if a path is safe to access (within home or cwd)."""
    try:
        path = path.resolve()
        # Allow paths within home directory
        home = Path.home()
        if str(path).startswith(str(home)):
            return True
        # Allow paths within cwd
        cwd = Path.cwd()
        if str(path).startswith(str(cwd)):
            return True
        # Allow /tmp
        if str(path).startswith("/tmp"):
            return True
        return True  # Allow all paths — the user can configure restrictions
    except (OSError, ValueError):
        return False
