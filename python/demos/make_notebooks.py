"""Generate Jupyter notebooks from the demo scripts.

The demos are plain Python scripts divided by banner comments of the form

    # ==========...
    #  1. Section title
    # ==========...

Each banner starts a new code cell, and becomes a markdown heading above it,
so the notebook has the same structure as the script and stays in step with
it. Run this after editing any demo:

    python3 demos/make_notebooks.py

Requires nothing beyond the standard library; the notebooks are written as
JSON directly.
"""

import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BANNER = re.compile(
    r"^# =+\n#\s+(.+?)\n# =+\n", re.M)


# The scripts add the parent directory to sys.path so they can be run from a
# clone without installing. A notebook has no __file__, and the documented
# route is now an installed package, so the bootstrap is dropped here.
BOOTSTRAP = re.compile(
    r"^import sys, os\nsys\.path\.insert\([^\n]*__file__[^\n]*\)\n", re.M)

# One demo locates the bundled audio relative to __file__, which a notebook
# also lacks; notebooks run from this directory, so a relative path serves.
AUDIO_DIR = re.compile(
    r"^audio_dir = Path\(__file__\)\.resolve\(\)\.parent\.parent / 'audio'$",
    re.M)


def cells_from(source):
    """Split a demo script into alternating markdown and code cells."""
    source = BOOTSTRAP.sub("", source)
    source = AUDIO_DIR.sub("audio_dir = Path('..') / 'audio'", source)
    # The module docstring becomes the notebook's opening markdown cell.
    doc = re.match(r'"""(.*?)"""\n', source, re.S)
    cells = []
    if doc:
        title, *rest = doc.group(1).strip().split("\n", 1)
        body = rest[0].strip() if rest else ""
        text = f"# {title}\n\n{body}" if body else f"# {title}"
        cells.append(markdown(text))
        source = source[doc.end():]

    parts = BANNER.split(source)
    preamble = parts[0].strip()
    if preamble:
        cells.append(code(preamble))
    for title, body in zip(parts[1::2], parts[2::2]):
        cells.append(markdown(f"## {title.strip()}"))
        body = body.strip()
        if body:
            cells.append(code(body))
    return cells


def cell_id(text):
    """A stable id derived from the cell's own text, as nbformat 4.5 wants."""
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def markdown(text):
    return {"cell_type": "markdown", "id": cell_id(text), "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "id": cell_id(text), "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    scripts = sorted(f for f in os.listdir(HERE)
                     if f.startswith("demo_") and f.endswith(".py"))
    for name in scripts:
        with open(os.path.join(HERE, name)) as fh:
            source = fh.read()
        out = os.path.join(HERE, name[:-3] + ".ipynb")
        with open(out, "w") as fh:
            json.dump(notebook(cells_from(source)), fh, indent=1)
            fh.write("\n")
        print(f"wrote {os.path.basename(out)}")


if __name__ == "__main__":
    main()
