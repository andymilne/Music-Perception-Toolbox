#!/usr/bin/env bash
# Build the routing figures. Each routing_r*.tex is a standalone document
# cropped to its own bounding box; the PNGs are what ARCHITECTURE.md embeds
# (GitHub does not render a PDF through Markdown image syntax), and the PDFs
# are kept for print use.
#
# Needs pdflatex (with the forest package) and pdftoppm (poppler).
set -euo pipefail
cd "$(dirname "$0")"
for n in 1 2 3 4; do
    pdflatex -interaction=nonstopmode -halt-on-error "routing_r$n.tex" >/dev/null
    pdftoppm -r 200 -png -singlefile "routing_r$n.pdf" "routing_r$n"
    echo "routing_r$n: $(pdfinfo routing_r$n.pdf | awk '/Page size/{print $3, $4, $5}')"
done
rm -f routing_r*.aux routing_r*.log
