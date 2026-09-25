"""jmm_data.py — the three works analysed in the JMM article, as MAET inputs.

The demos in this folder reproduce the worked examples of the article
(Milne, "Multi-attribute expectation tensors: smooth densities for
modelling musical structure, complexity, and similarity", *Journal of
Mathematics and Music*) and
its Online Supplement. This module holds the data they share, so that
each demo starts from the same encoding the article used:

* :func:`bwv347_notes` — Bach, *Ich dank dir, lieber Herre* (BWV 347),
  played through with the bars 1–4 repeat expanded, as an attribute
  table read from ``data/bwv347.musicxml`` with ``mpt.read_score`` (the
  article used the same score from the music21 corpus; the two encodings
  agree to the note). ``mpt.grid_attr_table`` samples it on whatever
  grid an analysis wants.
* :mod:`piano_phase` (a separate module beside this one) — Reich,
  *Piano Phase*: the twelve-note cell, the
  two voices rendered with the article's pulse and accelerando shape
  (smoothstep), a uniform hold/accelerando schedule standing in for the
  article's recording-transcribed one, and the phase as a function of
  time.
* :func:`derivations` — Ren, Rammos, and Rohrmeier's (2024)
  rule-labelled derivations of the Jazz Harmony Treebank, as a table of
  one row per path position, read from ``data/ParseTrees.json``. Not
  part of the toolbox distribution either; the function says where to
  fetch it.
* :func:`acknowledgement` — Coltrane, *Acknowledgement*: the solo, as
  an attribute table read from ``data/AwakeningSolo.mid`` with
  ``mpt.read_score``. The transcription is not part of the toolbox
  distribution; place your own MIDI transcription at that path.
"""
from __future__ import annotations

import os

import numpy as np

import mpt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GRID_STEP_QN = 0.25          # sixteenth note: the smallest value in BWV 347


# ---------------------------------------------------------------------------
#  Bach, BWV 347
# ---------------------------------------------------------------------------

def bwv347_notes():
    """The attribute table of the played-through chorale (``mpt.read_score``)."""
    return mpt.read_score(os.path.join(DATA_DIR, "bwv347.musicxml"))


def bwv347_fermata_spans():
    """``(start, end)`` quarter-note spans of the fermata-bearing notes of
    the played-through chorale, from the ``fermata`` column of the attribute
    table. Analysis 1.3 raises the weight of every eighth-note event under
    a fermata by half."""
    t = bwv347_notes()
    f = t["fermata"].to_numpy()
    on = t["onset_beats"].to_numpy()[f]
    dur = t["duration_beats"].to_numpy()[f]
    return sorted({(float(a), float(a + d)) for a, d in zip(on, dur)})


# ---------------------------------------------------------------------------
#  Coltrane, Acknowledgement
# ---------------------------------------------------------------------------

def acknowledgement(path=None):
    """The attribute table of the melody (``mpt.read_score``), one row per
    note, in onset order, from a monophonic MIDI transcription (default
    ``data/AwakeningSolo.mid``)."""
    path = path or os.path.join(DATA_DIR, "AwakeningSolo.mid")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. The transcription of Acknowledgement is "
            f"not distributed with the toolbox; place a monophonic MIDI "
            f"transcription of the solo at that path (or pass its path).")
    t = mpt.read_score(path)
    return t.sort_values("onset_beats", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
#  Ren, Rammos, and Rohrmeier (2024): derivations of the Jazz Harmony Treebank
# ---------------------------------------------------------------------------

PARSE_URL = ("https://github.com/ren-zeng/formal-modeling-of-structural-"
             "repetition/blob/main/experiment/DataSet/Harmony/ParseTrees.json")


def _paths(tree):
    """The root-to-leaf rule paths of one derivation, one per surface chord.

    A node is ``[chord, rule, children]``; a leaf carries its chord and no
    rule. The terminating rule directly above a leaf ends every path and
    says nothing about structure, so it is dropped. Each path is returned
    with, for each of its positions, the number of surface chords that
    position's node governs (the leaves below it), and with its own
    chord's quality.
    """
    out = []

    def leaves(node):
        if node.get("tag") == "Leaf":
            return 1
        return sum(leaves(child) for child in node["contents"][2])

    def walk(node, path):
        if node.get("tag") == "Leaf":
            kept = [step for step in path if step[0] != "Term"]
            quality = " ".join(node["contents"].get("quality") or [])
            out.append(([label for label, _ in kept],
                        [governed for _, governed in kept], quality))
            return
        _, rule, children = node["contents"]
        label = rule["contents"] if isinstance(rule.get("contents"), str) \
            else rule.get("tag")
        for child in children:
            walk(child, path + [(label, leaves(node))])

    walk(tree, [])
    return out


def derivations(tunes=None, path=None):
    """The rule-labelled derivations as a table, one row per path position.

    Columns: ``tune``, ``chord`` (the surface chord's index within its
    tune), ``level`` (the position's depth, the root at 1), ``label``
    (the rule applied there), ``governed`` (the number of surface chords
    the position's node governs), and ``quality`` (the surface chord's
    quality as the corpus records it, a stack of thirds such as
    ``'Maj Min Min'``, the dominant seventh). Reading a derivation as a table of
    positions is what lets the demo bind them: the positions of one chord
    are consecutive rows sharing a ``chord``.

    ``tunes`` selects by name (the corpus prefixes them ``(Valid)``);
    the default reads all 150.
    """
    path = path or os.path.join(DATA_DIR, "ParseTrees.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. The derivations are not distributed with "
            f"the toolbox; download ParseTrees.json from {PARSE_URL} and "
            f"place it at that path (or pass its path).")
    import json
    with open(path, encoding="utf-8") as fh:
        corpus = dict(json.load(fh))
    wanted = list(corpus) if tunes is None else list(tunes)
    rows = []
    for tune in wanted:
        if tune not in corpus:
            raise KeyError(f"{tune!r} is not in the corpus; it holds "
                           f"{len(corpus)} tunes, named like "
                           f"{next(iter(corpus))!r}.")
        for chord, (labels, governed, quality) in enumerate(
                _paths(corpus[tune])):
            for level, (label, m) in enumerate(zip(labels, governed),
                                               start=1):
                rows.append((tune, chord, level, label, m, quality))
    import pandas as pd
    return pd.DataFrame(rows, columns=["tune", "chord", "level", "label",
                                       "governed", "quality"])
