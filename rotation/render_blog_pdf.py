#!/usr/bin/env python3
"""Render a Markdown blog to PDF with its generated table of contents."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import markdown
from weasyprint import CSS, HTML


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--css", type=Path, required=True)
    args = parser.parse_args()

    document = markdown.Markdown(
        extensions=[
            "extra",
            "fenced_code",
            "sane_lists",
            "tables",
            "toc",
        ],
        extension_configs={"toc": {"permalink": False}},
        output_format="html5",
    )
    body = document.convert(args.input.read_text())
    title = args.input.read_text().splitlines()[0].removeprefix("# ").strip()
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
</head>
<body>
  <nav id="TOC">{document.toc}</nav>
  {body}
</body>
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=page, base_url=str(args.input.parent)).write_pdf(
        str(args.output),
        stylesheets=[CSS(filename=str(args.css))],
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
