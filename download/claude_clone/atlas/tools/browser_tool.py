"""
Atlas Browser Tool — headless browser automation via httpx + HTML parsing.

Features:
- Page navigation and content retrieval
- Form extraction and submission
- Link discovery
- Screenshot capture via subprocess (using wkhtmltoimage or similar)
- Cookie/session management (in-memory)
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from atlas.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# In-memory cookie jar
# ---------------------------------------------------------------------------

class _CookieJar:
    """Simple in-memory cookie jar keyed by domain."""

    def __init__(self) -> None:
        self._cookies: Dict[str, List[Dict[str, str]]] = {}

    def set(self, domain: str, name: str, value: str, path: str = "/") -> None:
        domain = domain.lower()
        if domain not in self._cookies:
            self._cookies[domain] = []
        # Replace if exists
        for c in self._cookies[domain]:
            if c["name"] == name:
                c["value"] = value
                c["path"] = path
                return
        self._cookies[domain].append({"name": name, "value": value, "path": path})

    def get_header(self, domain: str) -> str:
        """Return a Cookie header string for the given domain."""
        domain = domain.lower()
        parts = []
        for d, cookies in self._cookies.items():
            if domain.endswith(d) or d.endswith(domain):
                for c in cookies:
                    parts.append(f"{c['name']}={c['value']}")
        return "; ".join(parts)

    def clear(self, domain: Optional[str] = None) -> None:
        if domain:
            self._cookies.pop(domain.lower(), None)
        else:
            self._cookies.clear()

    def list_all(self) -> Dict[str, List[Dict[str, str]]]:
        return dict(self._cookies)


_cookie_jar = _CookieJar()


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


class _LinkExtractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.links: List[Dict[str, str]] = []
        self._base = base_url

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href", "")
            text = attrs_dict.get("title", "")
            if href:
                full = urljoin(self._base, href)
                self.links.append({"url": full, "text": text})


class _FormExtractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.forms: List[Dict[str, Any]] = []
        self._base = base_url
        self._current_form: Optional[Dict[str, Any]] = None
        self._in_textarea = False
        self._textarea_buf: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attrs_dict = dict(attrs)
        if tag == "form":
            action = attrs_dict.get("action", "")
            method = attrs_dict.get("method", "GET").upper()
            self._current_form = {
                "action": urljoin(self._base, action),
                "method": method,
                "fields": [],
            }
        elif tag == "input" and self._current_form is not None:
            input_type = attrs_dict.get("type", "text").lower()
            name = attrs_dict.get("name", "")
            value = attrs_dict.get("value", "")
            if name:
                self._current_form["fields"].append({
                    "name": name,
                    "type": input_type,
                    "value": value,
                })
        elif tag == "textarea" and self._current_form is not None:
            self._in_textarea = True
            self._textarea_buf = []
            name = attrs_dict.get("name", "")
            if name:
                self._current_form["fields"].append({
                    "name": name,
                    "type": "textarea",
                    "value": "",
                })

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._current_form is not None:
            self.forms.append(self._current_form)
            self._current_form = None
        elif tag == "textarea" and self._in_textarea:
            self._in_textarea = False
            # Update the last textarea field
            if self._current_form and self._current_form["fields"]:
                for f in reversed(self._current_form["fields"]):
                    if f["type"] == "textarea":
                        f["value"] = "".join(self._textarea_buf).strip()
                        break

    def handle_data(self, data: str) -> None:
        if self._in_textarea:
            self._textarea_buf.append(data)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def atlas_browser_navigate(url: str) -> str:
    """Navigate to a URL and return the page content summary.

    param url (str): — URL to navigate to.
    """
    try:
        import httpx
    except ImportError:
        return "Error: httpx is required. Install with: pip install httpx"

    try:
        domain = urlparse(url).netloc
        headers = dict(_HEADERS)
        cookie_header = _cookie_jar.get_header(domain)
        if cookie_header:
            headers["Cookie"] = cookie_header

        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()

        # Store any Set-Cookie headers
        for cookie_line in resp.headers.get_list("set-cookie"):
            parts = cookie_line.split(";")[0]
            if "=" in parts:
                name, value = parts.split("=", 1)
                _cookie_jar.set(domain, name.strip(), value.strip())

        # Extract title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", resp.text, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else "(no title)"

        # Extract links
        link_ext = _LinkExtractor(url)
        try:
            link_ext.feed(resp.text)
        except Exception:
            pass

        # Extract forms
        form_ext = _FormExtractor(url)
        try:
            form_ext.feed(resp.text)
        except Exception:
            pass

        # Extract text content (first 5000 chars)
        text_ext = _HTMLTextExtractor_for_browser()
        try:
            text_ext.feed(resp.text)
        except Exception:
            pass
        text = text_ext.get_text()
        if len(text) > 5000:
            text = text[:5000] + "\n[... more content available]"

        parts = [
            f"Title: {title}",
            f"URL: {resp.url}",
            f"Status: {resp.status_code}",
            f"Links found: {len(link_ext.links)}",
            f"Forms found: {len(form_ext.forms)}",
        ]
        if text:
            parts.append(f"\n--- Content ---\n{text}")

        return "\n".join(parts)

    except Exception as e:
        return f"Error navigating to {url}: {e}"


class _HTMLTextExtractor_for_browser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._result: List[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "noscript", "svg", "head"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript", "svg", "head"):
            self._skip = False
        if tag in ("p", "div", "h1", "h2", "h3", "br", "li"):
            self._result.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            t = data.strip()
            if t:
                self._result.append(t + " ")

    def get_text(self) -> str:
        raw = "".join(self._result)
        return re.sub(r"\s+", " ", raw).strip()


async def atlas_browser_get_links(url: str) -> str:
    """Extract all links from a page.

    param url (str): — URL to extract links from.
    """
    try:
        import httpx
    except ImportError:
        return "Error: httpx is required."

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=_HEADERS)
            resp.raise_for_status()

        ext = _LinkExtractor(str(resp.url))
        ext.feed(resp.text)

        if not ext.links:
            return f"No links found on {url}"

        lines = [f"Links on {resp.url} ({len(ext.links)} total):"]
        for i, link in enumerate(ext.links[:100], 1):
            label = link["text"] or "(no text)"
            lines.append(f"  {i}. {label} — {link['url']}")
        if len(ext.links) > 100:
            lines.append(f"  ... and {len(ext.links) - 100} more")
        return "\n".join(lines)

    except Exception as e:
        return f"Error extracting links: {e}"


async def atlas_browser_get_forms(url: str) -> str:
    """Extract all forms and their fields from a page.

    param url (str): — URL to extract forms from.
    """
    try:
        import httpx
    except ImportError:
        return "Error: httpx is required."

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=_HEADERS)
            resp.raise_for_status()

        ext = _FormExtractor(str(resp.url))
        ext.feed(resp.text)

        if not ext.forms:
            return f"No forms found on {url}"

        lines = [f"Forms on {resp.url} ({len(ext.forms)} total):"]
        for i, form in enumerate(ext.forms, 1):
            lines.append(f"\n  Form {i}: {form['method']} {form['action']}")
            for field in form["fields"]:
                lines.append(f"    - {field['name']} ({field['type']}): '{field['value']}'")

        return "\n".join(lines)

    except Exception as e:
        return f"Error extracting forms: {e}"


async def atlas_browser_submit_form(
    url: str,
    form_index: int = 0,
    field_values: str = "{}",
) -> str:
    """Submit a form on a page with given field values.

    param url (str): — URL of the page containing the form.
    param form_index (int): — Index of the form to submit (0-based). Default: 0.
    param field_values (str): — JSON string of field_name: value pairs to fill.
    """
    try:
        import httpx
    except ImportError:
        return "Error: httpx is required."

    try:
        values = json.loads(field_values) if isinstance(field_values, str) else field_values
    except json.JSONDecodeError:
        return f"Error: field_values must be valid JSON. Got: {field_values}"

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=_HEADERS)
            resp.raise_for_status()

        ext = _FormExtractor(str(resp.url))
        ext.feed(resp.text)

        if form_index >= len(ext.forms):
            return f"Error: Form index {form_index} out of range ({len(ext.forms)} forms found)"

        form = ext.forms[form_index]

        # Build form data
        data = {}
        for field in form["fields"]:
            if field["name"] in values:
                data[field["name"]] = str(values[field["name"]])
            elif field["value"]:
                data[field["name"]] = field["value"]

        # Submit
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                if form["method"] == "POST":
                    resp = await client.post(form["action"], data=data, headers=_HEADERS)
                else:
                    resp = await client.get(form["action"], params=data, headers=_HEADERS)
                resp.raise_for_status()
        except Exception as submit_err:
            return f"Error submitting form: {submit_err}"

        # Extract text from response
        text_ext = _HTMLTextExtractor_for_browser()
        try:
            text_ext.feed(resp.text)
        except Exception:
            pass
        text = text_ext.get_text()
        if len(text) > 5000:
            text = text[:5000] + "\n[...]"

        return f"Form submitted to {form['action']}\nStatus: {resp.status_code}\n\n{text or '(empty response)'}"

    except Exception as e:
        return f"Error: {e}"


async def atlas_browser_screenshot(url: str, output_path: str = "") -> str:
    """Capture a screenshot of a web page (requires wkhtmltoimage).

    param url (str): — URL to screenshot.
    param output_path (str): — Path to save the screenshot. Default: auto-generated.
    """
    if not output_path:
        output_path = str(Path(tempfile.gettempdir()) / f"screenshot_{int(asyncio.get_event_loop().time())}.png")

    try:
        proc = await asyncio.create_subprocess_exec(
            "wkhtmltoimage", "--quality", "80", "--width", "1280", url, output_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            if "command not found" in err or proc.returncode == 127:
                return "Error: wkhtmltoimage not found. Install with: apt install wkhtmltopdf"
            return f"Error taking screenshot: {err}"

        if Path(output_path).exists():
            size = Path(output_path).stat().st_size
            return f"Screenshot saved to {output_path} ({size:,} bytes)"
        return f"Screenshot command completed but file not found at {output_path}"

    except asyncio.TimeoutError:
        return "Error: Screenshot timed out"
    except Exception as e:
        return f"Error taking screenshot: {e}"


async def atlas_browser_cookies(domain: str = "") -> str:
    """List or manage cookies.

    param domain (str): — Domain to filter cookies. Empty string lists all.
    """
    all_cookies = _cookie_jar.list_all()
    if not all_cookies:
        return "No cookies stored."

    lines = ["Stored cookies:"]
    for d, cookies in sorted(all_cookies.items()):
        if domain and domain.lower() not in d:
            continue
        for c in cookies:
            lines.append(f"  {d}: {c['name']}={c['value'][:30]}{'...' if len(c['value'])>30 else ''}")

    return "\n".join(lines) if len(lines) > 1 else "No matching cookies."


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="atlas_browser_navigate",
    func=atlas_browser_navigate,
    description="Navigate to a URL and return page content summary with links and forms.",
    toolset="browser",
)

ToolRegistry.instance().register(
    name="atlas_browser_get_links",
    func=atlas_browser_get_links,
    description="Extract all hyperlinks from a web page.",
    toolset="browser",
)

ToolRegistry.instance().register(
    name="atlas_browser_get_forms",
    func=atlas_browser_get_forms,
    description="Extract all forms and their input fields from a web page.",
    toolset="browser",
)

ToolRegistry.instance().register(
    name="atlas_browser_submit_form",
    func=atlas_browser_submit_form,
    description="Fill and submit a form on a web page with custom field values.",
    toolset="browser",
)

ToolRegistry.instance().register(
    name="atlas_browser_screenshot",
    func=atlas_browser_screenshot,
    description="Capture a screenshot of a web page (requires wkhtmltoimage).",
    toolset="browser",
)

ToolRegistry.instance().register(
    name="atlas_browser_cookies",
    func=atlas_browser_cookies,
    description="List stored cookies for browser sessions.",
    toolset="browser",
)
