"""
Atlas Canvas Renderer — Renders canvas state to various output formats.

Provides rendering to HTML, Markdown, terminal (ANSI), JSON, and diff formats.
Includes a flexbox-like layout engine for positioning canvas elements.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class RenderFormat(Enum):
    """Supported render output formats."""
    HTML = "html"
    MARKDOWN = "markdown"
    TERMINAL = "terminal"
    JSON = "json"
    DIFF = "diff"


class Alignment(Enum):
    """Text alignment."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"


class FlexDirection(Enum):
    """Flex layout direction."""
    ROW = "row"
    COLUMN = "column"
    ROW_REVERSE = "row-reverse"
    COLUMN_REVERSE = "column-reverse"


class FlexWrap(Enum):
    """Flex wrapping behavior."""
    NO_WRAP = "nowrap"
    WRAP = "wrap"
    WRAP_REVERSE = "wrap-reverse"


class Overflow(Enum):
    """Overflow behavior."""
    VISIBLE = "visible"
    HIDDEN = "hidden"
    SCROLL = "scroll"
    AUTO = "auto"


# ANSI color codes for terminal rendering
ANSI_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "underline": "\033[4m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bg_black": "\033[40m",
    "bg_red": "\033[41m",
    "bg_green": "\033[42m",
    "bg_yellow": "\033[43m",
    "bg_blue": "\033[44m",
    "bg_magenta": "\033[45m",
    "bg_cyan": "\033[46m",
    "bg_white": "\033[47m",
    "border": "\033[90m",
}


@dataclass
class LayoutConstraints:
    """Layout constraints for an element."""
    min_width: int = 0
    max_width: int = 9999
    min_height: int = 0
    max_height: int = 9999
    padding: int = 0
    margin: int = 0
    gap: int = 8


@dataclass
class LayoutResult:
    """Result of layout calculation."""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    computed_style: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "computed_style": self.computed_style,
        }


@dataclass
class RenderedOutput:
    """Output from rendering a canvas."""
    format: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "content": self.content,
            "metadata": self.metadata,
        }


class LayoutEngine:
    """
    Flexbox-like layout engine for canvas elements.

    Calculates positions and sizes for canvas elements based on
    flexbox-inspired layout rules including direction, wrap,
    alignment, justification, and gap/spacing.

    Example:
        >>> engine = LayoutEngine(canvas_width=1024, canvas_height=768)
        >>> results = engine.layout(elements, direction=FlexDirection.COLUMN)
    """

    def __init__(
        self,
        canvas_width: int = 1024,
        canvas_height: int = 768,
        default_gap: int = 8,
        default_padding: int = 16,
    ) -> None:
        """
        Initialize the LayoutEngine.

        Args:
            canvas_width: Available canvas width.
            canvas_height: Available canvas height.
            default_gap: Default gap between elements.
            default_padding: Default padding around content.
        """
        self._canvas_width = canvas_width
        self._canvas_height = canvas_height
        self._default_gap = default_gap
        self._default_padding = default_padding

    def layout(
        self,
        elements: List[Dict[str, Any]],
        direction: FlexDirection = FlexDirection.COLUMN,
        wrap: FlexWrap = FlexWrap.NO_WRAP,
        justify: Alignment = Alignment.LEFT,
        align: Alignment = Alignment.STRETCH,
        gap: Optional[int] = None,
        padding: Optional[int] = None,
        container_width: Optional[int] = None,
        container_height: Optional[int] = None,
    ) -> List[LayoutResult]:
        """
        Calculate layout positions for a list of elements.

        Args:
            elements: List of element dictionaries.
            direction: Layout direction (row, column, etc.).
            wrap: Wrapping behavior.
            justify: Main axis alignment.
            align: Cross axis alignment.
            gap: Gap between elements.
            padding: Container padding.
            container_width: Override container width.
            container_height: Override container height.

        Returns:
            List of LayoutResult with computed positions.
        """
        gap = gap if gap is not None else self._default_gap
        padding = padding if padding is not None else self._default_padding
        container_width = container_width or self._canvas_width
        container_height = container_height or self._canvas_height

        available_width = container_width - 2 * padding
        available_height = container_height - 2 * padding

        if not elements:
            return []

        # Calculate element sizes
        element_sizes = []
        for el in elements:
            style = el.get("style", {})
            w = style.get("width", available_width)
            h = style.get("height", style.get("min_height", 40))

            # Flex grow/shrink
            flex_grow = style.get("flex_grow", style.get("flex-grow", 0))
            flex_shrink = style.get("flex_shrink", style.get("flex-shrink", 1))

            element_sizes.append({
                "element": el,
                "width": min(int(w), available_width),
                "height": int(h),
                "flex_grow": flex_grow,
                "flex_shrink": flex_shrink,
            })

        results: List[LayoutResult] = []

        if direction in (FlexDirection.COLUMN, FlexDirection.COLUMN_REVERSE):
            results = self._layout_vertical(
                element_sizes, padding, gap, available_width,
                available_height, justify, align,
                reverse=(direction == FlexDirection.COLUMN_REVERSE),
            )
        else:
            results = self._layout_horizontal(
                element_sizes, padding, gap, available_width,
                available_height, justify, align, wrap,
                reverse=(direction == FlexDirection.ROW_REVERSE),
            )

        return results

    def _layout_vertical(
        self,
        element_sizes: List[Dict[str, Any]],
        padding: int,
        gap: int,
        available_width: int,
        available_height: int,
        justify: Alignment,
        align: Alignment,
        reverse: bool = False,
    ) -> List[LayoutResult]:
        """Layout elements vertically."""
        total_height = sum(e["height"] for e in element_sizes)
        total_gaps = gap * (len(element_sizes) - 1) if len(element_sizes) > 1 else 0
        remaining = available_height - total_height - total_gaps

        # Apply flex grow
        growable = [e for e in element_sizes if e["flex_grow"] > 0]
        if growable and remaining > 0:
            total_grow = sum(e["flex_grow"] for e in growable)
            for e in element_sizes:
                if e["flex_grow"] > 0:
                    extra = int(remaining * e["flex_grow"] / total_grow)
                    e["height"] += extra

        # Calculate positions
        results = []
        current_y = padding

        if justify == Alignment.CENTER:
            current_y = padding + max(0, remaining) // 2
        elif justify == Alignment.RIGHT:
            current_y = padding + max(0, remaining)

        for el_info in element_sizes:
            el = el_info["element"]
            style = el.get("style", {})
            el_width = style.get("width", available_width)

            if align == Alignment.CENTER:
                x = padding + (available_width - min(int(el_width), available_width)) // 2
            elif align == Alignment.RIGHT:
                x = padding + available_width - min(int(el_width), available_width)
            else:
                x = padding

            results.append(LayoutResult(
                x=x,
                y=current_y,
                width=min(int(el_width), available_width),
                height=el_info["height"],
                computed_style={"position": "relative", "display": "flex"},
            ))

            current_y += el_info["height"] + gap

        if reverse:
            results.reverse()

        return results

    def _layout_horizontal(
        self,
        element_sizes: List[Dict[str, Any]],
        padding: int,
        gap: int,
        available_width: int,
        available_height: int,
        justify: Alignment,
        align: Alignment,
        wrap: FlexWrap,
        reverse: bool = False,
    ) -> List[LayoutResult]:
        """Layout elements horizontally with optional wrapping."""
        results: List[LayoutResult] = []
        current_x = padding
        current_y = padding
        row_height = 0

        for el_info in element_sizes:
            el = el_info["element"]
            style = el.get("style", {})
            el_width = min(int(style.get("width", available_width)), available_width)
            el_height = int(style.get("height", 40))

            # Check if we need to wrap
            if wrap != FlexWrap.NO_WRAP and current_x + el_width > padding + available_width:
                current_x = padding
                current_y += row_height + gap
                row_height = 0

            if align == Alignment.CENTER:
                y = current_y + (max(row_height, el_height) - el_height) // 2
            elif align == Alignment.STRETCH:
                el_height = max(el_height, row_height) if row_height > 0 else el_height
                y = current_y
            else:
                y = current_y

            results.append(LayoutResult(
                x=current_x,
                y=y,
                width=el_width,
                height=el_height,
                computed_style={"position": "relative"},
            ))

            current_x += el_width + gap
            row_height = max(row_height, el_height)

        # Justify main axis
        if justify == Alignment.CENTER and results:
            total_row_width = sum(r.width for r in results)
            total_gaps = gap * (len(results) - 1)
            offset = (available_width - total_row_width - total_gaps) // 2
            for r in results:
                r.x += max(0, offset)
        elif justify == Alignment.RIGHT and results:
            total_row_width = sum(r.width for r in results)
            total_gaps = gap * (len(results) - 1)
            offset = available_width - total_row_width - total_gaps
            for r in results:
                r.x += max(0, offset)

        if reverse:
            results.reverse()

        return results


class CanvasRenderer:
    """
    Renders canvas state to various output formats.

    Supports HTML, Markdown, terminal (ANSI), JSON, and diff rendering.
    Uses the LayoutEngine for element positioning.

    Example:
        >>> renderer = CanvasRenderer()
        >>> html = renderer.render_html(canvas_state)
        >>> md = renderer.render_markdown(canvas_state)
        >>> terminal = renderer.render_terminal(canvas_state)
    """

    def __init__(
        self,
        theme: str = "default",
        indent_size: int = 2,
        max_terminal_width: int = 80,
        colorize: bool = True,
    ) -> None:
        """
        Initialize the CanvasRenderer.

        Args:
            theme: Rendering theme (default, dark, compact).
            indent_size: Indentation for nested elements.
            max_terminal_width: Maximum width for terminal output.
            colorize: Whether to use ANSI colors in terminal output.
        """
        self._theme = theme
        self._indent = " " * indent_size
        self._max_width = max_terminal_width
        self._colorize = colorize
        self._layout_engine = LayoutEngine()

    def render_html(
        self,
        canvas_state: Dict[str, Any],
        include_styles: bool = True,
        interactive: bool = False,
    ) -> str:
        """
        Render canvas state to HTML.

        Args:
            canvas_state: Canvas state dictionary.
            include_styles: Include inline styles.
            interactive: Include JavaScript for interactivity.

        Returns:
            HTML string.
        """
        name = canvas_state.get("name", "canvas")
        width = canvas_state.get("width", 1024)
        height = canvas_state.get("height", 768)
        bg = canvas_state.get("background_color", "#ffffff")
        title = canvas_state.get("title", name)
        elements = canvas_state.get("elements", [])

        # Compute layout
        layout_engine = LayoutEngine(canvas_width=width, canvas_height=height)
        layout_results = layout_engine.layout(elements)

        # Build element lookup
        layout_map = {}
        for i, lr in enumerate(layout_results):
            if i < len(elements):
                layout_map[id(elements[i])] = lr

        # Generate HTML
        html_parts = [
            f'<!DOCTYPE html>',
            f'<html lang="en">',
            f'<head>',
            f'  <meta charset="UTF-8">',
            f'  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f'  <title>{self._escape_html(title)}</title>',
        ]

        if include_styles:
            html_parts.extend([
                f'  <style>',
                f'    .canvas-container {{',
                f'      width: {width}px;',
                f'      height: {height}px;',
                f'      background-color: {bg};',
                f'      overflow: auto;',
                f'      position: relative;',
                f'      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;',
                f'    }}',
                f'    .canvas-element {{',
                f'      position: absolute;',
                f'      box-sizing: border-box;',
                f'    }}',
                f'    .canvas-text {{ padding: 8px; word-wrap: break-word; }}',
                f'    .canvas-image {{ object-fit: contain; }}',
                f'    .canvas-code {{',
                f'      font-family: "SF Mono", "Fira Code", monospace;',
                f'      padding: 12px;',
                f'      background: #f5f5f5;',
                f'      border-radius: 4px;',
                f'      overflow-x: auto;',
                f'      white-space: pre;',
                f'    }}',
                f'    .canvas-table {{ border-collapse: collapse; width: 100%; }}',
                f'    .canvas-table th, .canvas-table td {{',
                f'      border: 1px solid #ddd;',
                f'      padding: 8px;',
                f'      text-align: left;',
                f'    }}',
                f'    .canvas-table th {{ background: #f0f0f0; font-weight: 600; }}',
                f'    .canvas-progress {{',
                f'      background: #e0e0e0;',
                f'      border-radius: 4px;',
                f'      overflow: hidden;',
                f'      height: 8px;',
                f'    }}',
                f'    .canvas-progress-bar {{',
                f'      background: linear-gradient(90deg, #4CAF50, #8BC34A);',
                f'      height: 100%;',
                f'      transition: width 0.3s ease;',
                f'    }}',
                f'    .canvas-button {{',
                f'      padding: 8px 16px;',
                f'      background: #1976D2;',
                f'      color: white;',
                f'      border: none;',
                f'      border-radius: 4px;',
                f'      cursor: pointer;',
                f'      font-size: 14px;',
                f'    }}',
                f'    .canvas-button:hover {{ background: #1565C0; }}',
                f'  </style>',
            ])

        html_parts.append(f'</head>')
        html_parts.append(f'<body>')
        html_parts.append(f'  <div class="canvas-container" id="{name}">')

        # Render each element
        for i, el in enumerate(elements):
            html_parts.append(self._render_element_html(el, layout_map.get(id(el))))

        html_parts.append(f'  </div>')

        if interactive:
            html_parts.extend([
                f'  <script>',
                f'    // A2UI WebSocket connection',
                f'    const ws = new WebSocket(`ws://${location.host}/ws/canvas/${name}`);',
                f'    ws.onmessage = (event) => {{',
                f'      const data = JSON.parse(event.data);',
                f'      if (data.type === "canvas_update") {{',
                f'        // Handle real-time updates',
                f'        console.log("Canvas update:", data);',
                f'      }}',
                f'    }};',
                f'  </script>',
            ])

        html_parts.append(f'</body></html>')

        return "\n".join(html_parts)

    def _render_element_html(
        self, element: Dict[str, Any], layout: Optional[LayoutResult] = None,
    ) -> str:
        """Render a single element to HTML."""
        el_type = element.get("type", "text")
        content = element.get("content", "")
        el_id = element.get("_id", f"el_{hash(content) % 10000}")
        style = element.get("style", {})

        # Position from layout
        pos_style = ""
        if layout:
            pos_style = f'style="left:{layout.x}px;top:{layout.y}px;width:{layout.width}px;height:{layout.height}px;"'

        # Additional inline styles
        custom_style = ""
        for key, val in style.items():
            css_key = key.replace("_", "-")
            custom_style += f"{css_key}:{val};"

        if custom_style:
            if pos_style:
                pos_style = pos_style.replace('style="', f'style="{custom_style}')
            else:
                pos_style = f'style="{custom_style}"'

        escaped_content = self._escape_html(str(content))

        if el_type == "text":
            text_style = style.get("font_size", "") or style.get("color", "")
            return f'    <div class="canvas-element canvas-text" id="{el_id}" {pos_style}>{escaped_content}</div>'

        elif el_type == "image":
            src = content
            alt = element.get("alt", element.get("description", "Image"))
            return f'    <img class="canvas-element canvas-image" id="{el_id}" src="{src}" alt="{self._escape_html(alt)}" {pos_style}>'

        elif el_type == "code":
            language = element.get("language", "")
            lang_attr = f'data-language="{language}"' if language else ""
            return f'    <pre class="canvas-element canvas-code" id="{el_id}" {lang_attr} {pos_style}>{escaped_content}</pre>'

        elif el_type == "table":
            return self._render_table_html(element, el_id, pos_style)

        elif el_type == "chart":
            return f'    <div class="canvas-element" id="{el_id}" {pos_style}><!-- Chart: {escaped_content} --></div>'

        elif el_type == "progress":
            value = element.get("value", 0)
            max_val = element.get("max", 100)
            percent = min(100, max(0, int(value / max_val * 100))) if max_val > 0 else 0
            label = element.get("label", "")
            return (
                f'    <div class="canvas-element" id="{el_id}" {pos_style}>'
                f'      {self._escape_html(label)} {percent}%'
                f'      <div class="canvas-progress">'
                f'        <div class="canvas-progress-bar" style="width:{percent}%"></div>'
                f'      </div>'
                f'    </div>'
            )

        elif el_type == "button":
            label = element.get("label", content)
            action = element.get("action", "")
            onclick = f' onclick="a2ui_action(\'{action}\')"' if action else ""
            return f'    <button class="canvas-element canvas-button" id="{el_id}"{onclick} {pos_style}>{self._escape_html(label)}</button>'

        elif el_type == "form":
            return f'    <form class="canvas-element" id="{el_id}" {pos_style}>{escaped_content}</form>'

        else:
            return f'    <div class="canvas-element" id="{el_id}" {pos_style}>{escaped_content}</div>'

    def _render_table_html(self, element: Dict[str, Any], el_id: str, pos_style: str) -> str:
        """Render a table element to HTML."""
        headers = element.get("headers", element.get("columns", []))
        rows = element.get("rows", element.get("data", []))

        parts = [f'    <table class="canvas-element canvas-table" id="{el_id}" {pos_style}>']

        if headers:
            parts.append("      <thead><tr>")
            for h in headers:
                parts.append(f"        <th>{self._escape_html(str(h))}</th>")
            parts.append("      </tr></thead>")

        if rows:
            parts.append("      <tbody>")
            for row in rows:
                parts.append("        <tr>")
                cells = row if isinstance(row, list) else [row]
                for cell in cells:
                    parts.append(f"          <td>{self._escape_html(str(cell))}</td>")
                parts.append("        </tr>")
            parts.append("      </tbody>")

        parts.append("    </table>")
        return "\n".join(parts)

    def render_markdown(self, canvas_state: Dict[str, Any]) -> str:
        """
        Render canvas state to Markdown.

        Args:
            canvas_state: Canvas state dictionary.

        Returns:
            Markdown string.
        """
        title = canvas_state.get("title", canvas_state.get("name", "Canvas"))
        elements = canvas_state.get("elements", [])

        parts = [f"# {title}", ""]

        for el in elements:
            el_type = el.get("type", "text")
            content = el.get("content", "")

            if el_type == "text":
                parts.append(str(content))
                parts.append("")
            elif el_type == "image":
                alt = el.get("alt", el.get("description", "Image"))
                parts.append(f"![{alt}]({content})")
                parts.append("")
            elif el_type == "code":
                language = el.get("language", "")
                parts.append(f"```{language}")
                parts.append(str(content))
                parts.append("```")
                parts.append("")
            elif el_type == "table":
                parts.append(self._render_table_markdown(el))
                parts.append("")
            elif el_type == "progress":
                value = el.get("value", 0)
                max_val = el.get("max", 100)
                label = el.get("label", "Progress")
                percent = int(value / max_val * 100) if max_val > 0 else 0
                bar_len = 20
                filled = int(bar_len * percent / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                parts.append(f"**{label}**: {bar} {percent}%")
                parts.append("")
            elif el_type == "button":
                label = el.get("label", content)
                parts.append(f"[{label}]")
                parts.append("")
            elif el_type == "heading":
                level = el.get("level", 2)
                prefix = "#" * level
                parts.append(f"{prefix} {content}")
                parts.append("")
            elif el_type == "divider":
                parts.append("---")
                parts.append("")
            elif el_type == "list":
                items = el.get("items", content.split("\n") if isinstance(content, str) else [])
                ordered = el.get("ordered", False)
                for j, item in enumerate(items):
                    if ordered:
                        parts.append(f"{j + 1}. {item}")
                    else:
                        parts.append(f"- {item}")
                parts.append("")
            else:
                parts.append(str(content))
                parts.append("")

        return "\n".join(parts)

    def _render_table_markdown(self, element: Dict[str, Any]) -> str:
        """Render a table element to Markdown."""
        headers = element.get("headers", element.get("columns", []))
        rows = element.get("rows", element.get("data", []))

        if not headers or not rows:
            return ""

        parts = []

        # Header row
        parts.append("| " + " | ".join(str(h) for h in headers) + " |")
        # Separator
        parts.append("| " + " | ".join("---" for _ in headers) + " |")
        # Data rows
        for row in rows:
            cells = row if isinstance(row, list) else [row]
            parts.append("| " + " | ".join(str(c) for c in cells) + " |")

        return "\n".join(parts)

    def render_terminal(
        self,
        canvas_state: Dict[str, Any],
        colorize: Optional[bool] = None,
    ) -> str:
        """
        Render canvas state for terminal output with ANSI formatting.

        Args:
            canvas_state: Canvas state dictionary.
            colorize: Whether to use ANSI colors. Uses default if None.

        Returns:
            Terminal-formatted string with ANSI codes.
        """
        use_color = colorize if colorize is not None else self._colorize
        title = canvas_state.get("title", canvas_state.get("name", "Canvas"))
        elements = canvas_state.get("elements", [])
        width = canvas_state.get("width", self._max_width)
        max_w = min(width, self._max_width)

        parts = []

        if use_color:
            parts.append(f"{ANSI_COLORS['bold']}{ANSI_COLORS['cyan']}{'─' * max_w}{ANSI_COLORS['reset']}")
            parts.append(f"{ANSI_COLORS['bold']}{ANSI_COLORS['cyan']}  {title}{ANSI_COLORS['reset']}")
            parts.append(f"{ANSI_COLORS['bold']}{ANSI_COLORS['cyan']}{'─' * max_w}{ANSI_COLORS['reset']}")
        else:
            parts.append("─" * max_w)
            parts.append(f"  {title}")
            parts.append("─" * max_w)

        parts.append("")

        for el in elements:
            el_type = el.get("type", "text")
            content = str(el.get("content", ""))
            style = el.get("style", {})

            if el_type == "heading":
                level = el.get("level", 2)
                if use_color:
                    prefix = "  " + ("#" * min(level, 4)) + " "
                    parts.append(f"{ANSI_COLORS['bold']}{ANSI_COLORS['yellow']}{prefix}{self._truncate(content, max_w)}{ANSI_COLORS['reset']}")
                else:
                    parts.append(f"  {'#' * min(level, 4)} {self._truncate(content, max_w)}")

            elif el_type == "text":
                lines = content.split("\n")
                for line in lines:
                    truncated = self._truncate(line, max_w - 4)
                    color = style.get("color", "")
                    if use_color and color in ANSI_COLORS:
                        parts.append(f"  {ANSI_COLORS[color]}{truncated}{ANSI_COLORS['reset']}")
                    elif use_color and style.get("bold"):
                        parts.append(f"  {ANSI_COLORS['bold']}{truncated}{ANSI_COLORS['reset']}")
                    else:
                        parts.append(f"  {truncated}")

            elif el_type == "code":
                lang = el.get("language", "")
                if use_color:
                    parts.append(f"  {ANSI_COLORS['bg_black']}{ANSI_COLORS['green']}  ┌ {lang}{ANSI_COLORS['reset']}")
                    for line in content.split("\n")[:10]:
                        parts.append(f"  {ANSI_COLORS['bg_black']}{ANSI_COLORS['green']}  │ {line[:max_w - 6]}{ANSI_COLORS['reset']}")
                    if content.count("\n") > 10:
                        parts.append(f"  {ANSI_COLORS['dim']}  ... ({content.count(chr(10)) - 10} more lines){ANSI_COLORS['reset']}")
                    parts.append(f"  {ANSI_COLORS['bg_black']}{ANSI_COLORS['green']}  └{ANSI_COLORS['reset']}")
                else:
                    parts.append(f"  [{lang}]")
                    for line in content.split("\n")[:10]:
                        parts.append(f"    {line[:max_w - 4]}")

            elif el_type == "table":
                parts.append(self._render_table_terminal(el, max_w, use_color))

            elif el_type == "progress":
                value = el.get("value", 0)
                max_val = el.get("max", 100)
                label = el.get("label", "Progress")
                percent = int(value / max_val * 100) if max_val > 0 else 0
                bar_len = min(30, max_w - len(label) - 10)
                filled = int(bar_len * percent / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                if use_color:
                    parts.append(f"  {ANSI_COLORS['bold']}{label}{ANSI_COLORS['reset']} [{ANSI_COLORS['green']}{bar}{ANSI_COLORS['reset']}] {percent}%")
                else:
                    parts.append(f"  {label} [{bar}] {percent}%")

            elif el_type == "button":
                label = el.get("label", content)
                if use_color:
                    parts.append(f"  {ANSI_COLORS['bg_blue']}{ANSI_COLORS['white']}  [ {label} ]{ANSI_COLORS['reset']}")
                else:
                    parts.append(f"  [ {label} ]")

            elif el_type == "divider":
                if use_color:
                    parts.append(f"  {ANSI_COLORS['border']}{'─' * (max_w - 4)}{ANSI_COLORS['reset']}")
                else:
                    parts.append(f"  {'─' * (max_w - 4)}")

            elif el_type == "list":
                items = el.get("items", content.split("\n"))
                ordered = el.get("ordered", False)
                for j, item in enumerate(items):
                    prefix = f"  {j + 1}." if ordered else "  •"
                    if use_color:
                        parts.append(f"  {ANSI_COLORS['cyan']}{prefix}{ANSI_COLORS['reset']} {item}")
                    else:
                        parts.append(f"{prefix} {item}")

            else:
                parts.append(f"  {self._truncate(content, max_w - 4)}")

            parts.append("")

        if use_color:
            parts.append(f"{ANSI_COLORS['border']}{'─' * max_w}{ANSI_COLORS['reset']}")
        else:
            parts.append("─" * max_w)

        return "\n".join(parts)

    def _render_table_terminal(
        self, element: Dict[str, Any], max_w: int, colorize: bool,
    ) -> str:
        """Render a table element for terminal output."""
        headers = element.get("headers", element.get("columns", []))
        rows = element.get("rows", element.get("data", []))

        if not headers:
            return ""

        # Calculate column widths
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            cells = row if isinstance(row, list) else [row]
            for i, cell in enumerate(cells):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        # Truncate to fit width
        total_width = sum(col_widths) + 3 * len(col_widths) + 1
        if total_width > max_w - 4:
            scale = (max_w - 4) / total_width
            col_widths = [max(3, int(w * scale)) for w in col_widths]

        parts = []
        sep = "+" + "+".join("─" * (w + 2) for w in col_widths) + "+"

        if colorize:
            parts.append(f"  {ANSI_COLORS['bold']}{sep}{ANSI_COLORS['reset']}")
            header_line = "|"
            for i, h in enumerate(headers):
                header_line += f" {str(h):<{col_widths[i]}} |"
            parts.append(f"  {ANSI_COLORS['bold']}{ANSI_COLORS['yellow']}{header_line}{ANSI_COLORS['reset']}")
            parts.append(f"  {ANSI_COLORS['bold']}{sep}{ANSI_COLORS['reset']}")
        else:
            parts.append(f"  {sep}")
            header_line = "|"
            for i, h in enumerate(headers):
                header_line += f" {str(h):<{col_widths[i]}} |"
            parts.append(f"  {header_line}")
            parts.append(f"  {sep}")

        for row in rows[:20]:  # Limit to 20 rows
            cells = row if isinstance(row, list) else [row]
            line = "|"
            for i in range(len(col_widths)):
                cell = str(cells[i]) if i < len(cells) else ""
                line += f" {cell:<{col_widths[i]}} |"
            parts.append(f"  {line}")

        parts.append(f"  {sep}")
        if len(rows) > 20:
            parts.append(f"  ... and {len(rows) - 20} more rows")

        return "\n".join(parts)

    def render_json(
        self,
        canvas_state: Dict[str, Any],
        pretty: bool = True,
    ) -> str:
        """
        Render canvas state to JSON.

        Args:
            canvas_state: Canvas state dictionary.
            pretty: Whether to pretty-print the JSON.

        Returns:
            JSON string.
        """
        indent = 2 if pretty else None
        return json.dumps(canvas_state, indent=indent, default=str, ensure_ascii=False)

    def render_diff(
        self,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
    ) -> str:
        """
        Render a diff between two canvas states.

        Args:
            old_state: Previous canvas state.
            new_state: New canvas state.

        Returns:
            Human-readable diff string.
        """
        parts = []
        name = new_state.get("name", old_state.get("name", "canvas"))

        parts.append(f"=== Canvas Diff: {name} ===")
        parts.append("")

        # Version diff
        old_ver = old_state.get("version", 0)
        new_ver = new_state.get("version", 0)
        if old_ver != new_ver:
            parts.append(f"  Version: {old_ver} → {new_ver}")
            parts.append("")

        # Element diff
        old_elements = old_state.get("elements", [])
        new_elements = new_state.get("elements", [])
        old_ids = {el.get("_id") for el in old_elements}
        new_ids = {el.get("_id") for el in new_elements}

        added = new_ids - old_ids
        removed = old_ids - new_ids
        common = old_ids & new_ids

        if added:
            parts.append(f"  + {len(added)} element(s) added:")
            for eid in added:
                el = next((e for e in new_elements if e.get("_id") == eid), {})
                el_type = el.get("type", "unknown")
                content = str(el.get("content", ""))[:50]
                parts.append(f"    + [{el_type}] {content}")
            parts.append("")

        if removed:
            parts.append(f"  - {len(removed)} element(s) removed:")
            for eid in removed:
                el = next((e for e in old_elements if e.get("_id") == eid), {})
                el_type = el.get("type", "unknown")
                content = str(el.get("content", ""))[:50]
                parts.append(f"    - [{el_type}] {content}")
            parts.append("")

        if common:
            modified = 0
            for eid in common:
                old_el = next((e for e in old_elements if e.get("_id") == eid), {})
                new_el = next((e for e in new_elements if e.get("_id") == eid), {})
                # Simple content comparison
                old_content = str(old_el.get("content", ""))
                new_content = str(new_el.get("content", ""))
                if old_content != new_content:
                    modified += 1
                    el_type = new_el.get("type", "unknown")
                    parts.append(f"    ~ [{el_type}] {eid}:")
                    parts.append(f"      - {old_content[:60]}")
                    parts.append(f"      + {new_content[:60]}")

            if modified:
                parts.insert(len(parts) - 1, f"  ~ {modified} element(s) modified:")
                parts.append("")

        # Style diff
        old_styles = old_state.get("styles", {})
        new_styles = new_state.get("styles", {})
        if old_styles != new_styles:
            parts.append("  Styles changed:")
            for key in set(list(old_styles.keys()) + list(new_styles.keys())):
                old_val = old_styles.get(key, "<none>")
                new_val = new_styles.get(key, "<none>")
                if old_val != new_val:
                    parts.append(f"    {key}: {old_val} → {new_val}")
            parts.append("")

        # Title diff
        old_title = old_state.get("title", "")
        new_title = new_state.get("title", "")
        if old_title != new_title:
            parts.append(f"  Title: '{old_title}' → '{new_title}'")
            parts.append("")

        if not any([added, removed, modified if common else False,
                    old_styles != new_styles, old_title != new_title,
                    old_ver != new_ver]):
            parts.append("  (no changes)")

        parts.append("")
        parts.append(f"=== End Diff: {name} ===")

        return "\n".join(parts)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text to maximum length with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."
