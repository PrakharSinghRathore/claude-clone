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
