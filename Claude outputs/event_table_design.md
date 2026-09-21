# The event table, the grid, and categorical encoding — design

Status: all eight steps of Section 11 are implemented (the readers return a
table; sustain, sostenuto, pitch bend, and the loudness controllers are
resolved at read; `buildMaet` admits an event empty on an attribute;
`gridEvents` samples a table on a grid; the conversion carries the three
categorical roles; `selectPreMaet` filters a pre-MAET; the readers carry more of what a
score holds; the documentation follows the code). Supersedes the note table of
`readScore` / `read_score` and the score path of `preMaetFromScore` /
`pre_maet_from_score` if accepted.

## 0. The interface, in full

The whole of what this design adds to the public surface:

    readScore(path)                -> table
    gridEvents(T, ...)             -> table
    preMaetFromScore(path|T, ...)  -> pre-MAET
    selectPreMaet(pm, ...)         -> pre-MAET

One new function for scores (`gridEvents`) and one for pre-MAETs
(`selectPreMaet`, named provisionally). `readScore` returns a standard
table rather than a bespoke struct; `preMaetFromScore` still takes a path,
so `readScore` is needed only when the table itself is wanted.

There is deliberately **no event-table type, and no metadata layer**. An
event table is a MATLAB `table` and a pandas `DataFrame`, nothing more.
Everything the toolbox needs to know about a column is either its name or
its type, both of which the host language already carries and the user
already knows how to read.

## 1. What is wrong now

`readScore` returns a struct of N-length numeric columns with two fields
that are not per note (`partNames`, `source`) bolted on. That object is not
a table, is not a standard type, and nothing outside the toolbox is
designed to parse it. Three specific consequences:

- **One column carries two meanings.** `channel` holds a MIDI channel on
  one path and a MusicXML voice on the other. Any operation that consults
  it reads a different quantity depending on where the file came from.
  Pedal resolution and re-strike damping both need the real channel, so
  this is not cosmetic.
- **Everything is numeric.** `fermata` is 1 or 0 rather than a logical;
  `part` is an integer whose names live in a separate field; there is no
  representation at all for a text label, so articulations, lyrics,
  dynamics, and any user-supplied label have nowhere to go.
- **The conversion knows about provenance.** `preMaetFromScore` takes a
  `'parts'` option and a fixed `'attributes'` vocabulary drawn from the
  score reader's own column names, so selection and encoding are entangled
  with parsing.

## 2. The event table

One row per event, one column per attribute. A row is a note; after
gridding it is a note at one grid point. Row granularity never varies
within a table.

### 2.1 Reserved names

Four column names are read by name where they are present:

    onset, duration, pitch, weight

Everything else — `channel`, `voice`, `part`, `staff`, articulations,
lyrics, timbre features, spatial coordinates — is ordinary vocabulary and
is usable on identical terms. There is no second class.

This dissolves the `channel` / `voice` overloading: the MIDI reader emits
`channel`, the MusicXML reader emits `voice`, each means what its name
says, and a user who wants to treat them alike says so.

### 2.2 The column type is the declaration

A numeric column is a numeric attribute. A categorical column is a
categorical attribute whose levels are its categories, in category order.
Nothing is stored beside the column, so nothing can fall out of step with
it, and a user reads a level set with `categories(T.voice)` or
`df["voice"].cat.categories` rather than through anything of ours.

Both languages' categorical types keep categories that no row uses, which
is the behaviour required here: selecting rows must not silently change a
level set. Two tables to be compared with one another must declare the
same categories in the same order, since otherwise one table's levels are
matched against another's. That is the single reproducibility rule, and it
is enforced by construction if both tables come from the same reader.

Anything constant across the rows — which file the rows came from, and the
like — belongs in the table's own description field
(`Properties.Description`, `df.attrs`), not in a column and not in an API
of ours.

### 2.3 Selection

The toolbox supplies no table filter. A real table already has one —
`T(T.part == "Violin I", :)`, `df[df.part == "Violin I"]` — better
documented than anything written here would be. Semantic selection is the
host language's job; the toolbox's only filter is the pre-MAET one of
Section 8.

### 2.4 Missing values

An attribute that an event does not have is a missing entry in its column
— `NaN` in a numeric column, `<undefined>` in a MATLAB categorical, `NA`
in a pandas one — not a sentinel value and not a row of a different shape.
A rest on a grid is a row carrying a grid position and a time with every
note column missing; a rest in one voice of a four-voice texture is the
same thing one row at a time.

This is what carries an event empty on some attributes and populated on
others (Section 9.1) into the pre-MAET, where a missing entry contributes
no element to that attribute's multiset: `K_{a,n}` counts the entries
present. The pre-MAET already pads with `NaN` at zero weight, so the two
representations agree.

Consequence for the aggregating roles of Section 7: they require every
gathered event to be fully populated, so a missing entry is where that
requirement fails, and where the conversion must say so rather than pad
silently.

### 2.5 Validation

Validation belongs to the function that depends on it. `gridEvents` and
`preMaetFromScore` check the columns they use — that `onset` and
`duration` are real and numeric, that `duration` and `weight` are
non-negative, that a column named in a role specification exists and is
categorical — and say which argument was at fault. There is no separate
construction step at which a table must be blessed.

## 3. Time

`duration`, not the time of the note-off. A duration is an attribute; a
stop time is a coordinate. Attribute translation adds an offset to every
value of a selected attribute, so under a time shift onsets move and
durations do not; storing stop times would require two columns to be
updated in step or the note lengths would be silently corrupted. Stop time
is one addition away wherever overlap or grid occupancy is computed.

Beats and seconds are both kept, as now, under names that say which is
which. Neither is derivable from the other without the tempo map when a
tempo change falls inside a note.

## 4. Pedal, and sounding duration

Sustain suspends the effect of note-offs, and this is resolved at read time
into the note data rather than left to the user. Two columns, both lengths
from the same onset:

- `duration` — note-on to its matching note-off, as recorded.
- `soundingDuration` — normally `max(noteOff, pedalRelease)`, truncated at
  a re-strike.

Which column feeds the conversion is the analyst's choice, exactly as the
JMM manuscript treats notated against sustained fermata chords (Analyses
1.1–1.2 take durations as notated; Analysis 1.3 handles the sustain by
weighting).

Conventions the resolution fixes, each to be documented:

- CC64 at or above 64 is down; half-pedalling is not representable and is
  collapsed.
- **Re-strike**: a pitch struck again on the same channel ends the earlier
  instance at the re-strike, so no overlap of one pitch on one channel
  arises. A pitch struck on a different channel sustains into an overlap,
  since channels may be different instruments. The rule therefore needs the
  real channel, and pedal state is likewise per channel.
- Sostenuto (CC66) holds only notes already down when depressed, so it is
  not handled by the same forward fill.
- A pedal never released before end of file terminates there.

Channel volume and expression likewise resolve into `weight` at read, under
the specification's approximately squared amplitude curve, rather than
becoming attributes of their own.

All other continuous controllers are **out of scope**. A column holding a
controller's value at onset looks as though it carries the information and
does not: a ramp within a held note is invisible in it. Sub-note variation
is declared out of scope rather than approximated. See Section 10 for the
extension that would admit it.

## 5. Pitch bend, microtonality, and MPE

Pitch bend resolves into `pitch` at read time, and `pitch` is therefore a
floating-point quantity, not a note number. This is not an expressive
nicety: bend is the standard way microtonal music is carried in MIDI, both
in the one-channel-per-note idiom and under MPE, so a reader that returns
integers returns the wrong notes.

- `pitch` carries the resolved value in the chosen scale. `noteNumber`
  keeps the note number as recorded, which is also what the re-strike and
  note-off matching rules use, since note identity is by note number and
  not by sounding pitch.
- Bend is 14-bit against a per-channel range set by RPN 0. The range
  default is 2 semitones outside MPE and 48 semitones for MPE member
  channels. An MPE Configuration Message (RPN 6 on channel 1 or 16)
  selects a zone and therefore selects the second default.
- Where bend is present but no RPN 0 and no MCM were seen, the reader
  warns, since a file tuned for one range read at the other is wrong by a
  factor of 24.
- **Ordering at the onset.** Exporters differ over whether the bend for a
  note is sent immediately before or immediately after its note-on, and
  within a tick the ordering carries no meaning. Sampling strictly at the
  onset therefore misses the tuning of some files entirely. The convention
  needs to be the most recent bend at or before the onset, or, where the
  channel has none, the first bend within a small look-ahead. The
  look-ahead size is a decision.
- Under MPE, bend continues through the note as a slide, so the resolved
  value is the pitch at onset. That is exactly right for static microtonal
  tuning, which is the dominant use, and is a documented reduction for
  glides. The extension of Section 10 is what would carry the whole
  contour.

## 6. The two granularities, and the gridder

The JMM manuscript requires a metrical grid: Analysis 1.1 casts BWV 347 as
a MAET "sampled on a sixteenth-note grid", N = 272 at 0.25 QN, with "held
notes replicat[ing] across the grid points they occupy". Analysis 1.2 uses
the same grid. This is not speculative.

`gridEvents` is a table-to-table transformation, so it composes with
selection in either order:

    readScore  ->  table  ->  [gridEvents]  ->  [select]  ->  preMaetFromScore

For score data — where there are no controller streams — the note table is
sufficient to determine every grid point, so the gridder needs nothing the
read has discarded.

Each grid point opens a slice, and a note belongs to every slice it
overlaps, so nothing falls between samples. The weight policy says what a
slice takes from a note overlapping it:

- **coverage** (default) — the fraction of the slice the note fills. This
  is Analysis 1.3's weighting, "the fraction of the eighth each note
  sounds".
- **presence** — the note's full weight in every slice it appears in at
  all, however briefly: which notes are here, rather than how much of the
  span each occupies. A membership reading: each slice records the set of
  what occurs in it, whatever the step, separating from coverage as the
  step grows relative to the notes -- at a bar-length step a slice holds
  the set of what occurs in that bar, where coverage holds a
  duration-weighted profile of it. It differs from coverage only where a
  note does not fill a slice, so on a grid at or finer than the shortest
  note the two agree. For what is sounding at a given moment the
  instrument is coverage on a fine grid, an instant having no duration.
- **item** — the fraction of the note in the slice, so that its weight is
  distributed over the slices it spans and it counts once in total. For
  attributes constant over the note this gives a density numerically
  identical to the ungridded one at `r = 1`, the kernel being linear in
  weight.

It adds a grid position column and a note identifier, so the grid table
collapses back to the note table by grouping and nothing is lost.
`duration` keeps meaning the note's duration; the slice length is a
property of the grid.

A metrical grid presupposes a beat map. Score data has one (`onsetBeats`);
raw performance MIDI may not, so the specification admits a clock grid in
seconds and fails loudly rather than assuming a tempo.

## 7. The conversion to a pre-MAET

### 7.1 The three categorical roles

The manuscript (Sec. 2.2 of the minimal article) names ordered encoding and
simplex coding, noting the separate-attribute form equivalent to the first.
Named for the structure each produces:

| role | outcome | event grain | requires |
| --- | --- | --- | --- |
| `separateAttributes` | several attributes, one per level | aggregating | exclusive levels, fixed cardinality, nominated value columns |
| `orderedMultiset` | one attribute, level by position (`[sym] = 0`, `r = K`) | aggregating | as above; admits `[rel] = 1` |
| `simplex` | one attribute, level by vertex (`r = V - 1`) | row-wise | the level list only |
| `drop` | nothing | unchanged | — |

Python spells these `separate_attributes`, `ordered_multiset`, `simplex`,
`drop`.

The first two carry the binding of value to level in the layout; the third
carries it in the tensor product of two attributes at an event, which is
why simplex-voice gives additive partial credit where voice-aware gives a
multiplicative AND (Analysis 1.2).

**Aggregating** means rows sharing a grouping key become one event whose
attributes are populated from them; **row-wise** means each row becomes one
event. The grouping key defaults to onset, which is exact on a grid and
needs a tolerance or an explicit column for performance data — a further
reason to grid a performance before converting.

The role is given in the conversion call, not attached to the column:
Analysis 1.2 converts one chorale three ways, which settles where it
belongs.

### 7.2 One aggregating category

Only one category may take an aggregating role against a given value set,
and it must be the one that individuates concurrent values. Voice does;
articulation does not, being a property of a note rather than a way of
telling simultaneous notes apart. Two aggregating categories give an
attribute set that can never be fully populated — `voice x articulation`
yields eight attributes of which at most four are ever filled, and which
four varies by event.

The aggregating category splits **every** per-note column, not only pitch:
voice splits pitch into four attributes and articulation into four
attributes, each carrying that voice's simplex coordinates. The event is
still the chord, every attribute is populated, and each note's articulation
stays bound to its own voice by the layout.

The warning, when a second category is given an aggregating role, names the
category already doing so and offers the two ways out: keep voice
aggregating and tag the second category within each voice slot; or give
voice the `simplex` role, yielding one event per note, at which point every
category is a tag and matching becomes additive partial credit. Both are
encodings Analysis 1.2 already contrasts.

### 7.3 What the conversion does not do

There is no ordinal-to-number role. If a category's levels have numeric
values, the column is not categorical — it is numeric recorded as labels —
and the fix belongs in the table, not the conversion, which would otherwise
be asserting a spacing on the user's behalf.

### 7.4 Absence

A note with no articulation has no natural point in a simplex. Default to
an explicit level, which stands equidistant from every marking and keeps K
constant; allow omission for cases such as lyrics, where a varying K is the
right reading.

### 7.5 Simplex construction

Unit edge length, already the toolbox's convention (the Supplement
specifies `sigma = 0.1` against unit-edge label simplices), which makes
`sigma` interpretable and so is not a user parameter. V = 2 is the
degenerate case at plus or minus one half. `simplexVertices` already exists
and is cited in the manuscript. Since the MAET depends only on distances,
the basis is immaterial to any result, but it is pinned to one
deterministic algorithm so that MATLAB and Python display the same
coordinates.

## 8. The pre-MAET filter

A single function operating on attributes and events only, knowing nothing
of the data's provenance. It selects attributes by index or group name and
events by index or by a predicate on values. It must refuse to split an
attribute group, since dropping one coordinate of a simplex projects it
onto an arbitrary subspace of itself and names no level at all.

Dropping attributes may leave numerically identical events. Merging them
with summed weights gives exactly the same MAET, the density being linear
in weight, so merging is an efficiency choice rather than a semantic one.
The one thing to check is any normalization dividing by an event count
rather than a weight sum.

## 9. Open decisions

1. **Empty grid slices — resolved: they are kept, and `buildMaet` must
   admit them.** A grid point with nothing sounding is an event empty on
   pitch while carrying a time, and an event empty on some attributes but
   not others has to be kept: windowing operates over one attribute and
   acts on another, so the populated attributes are doing work even where
   one is empty. Dropping such an event also breaks adjacency, since
   `bindEvents` and `differenceEvents` would then treat the events on
   either side of a rest as consecutive, and it destroys the property a
   grid exists to provide, that the event index is a uniform time index.

   Keeping them changes no value: an event with no admitted tuple on an
   attribute contributes no term to the joint density, hence none to the
   inner product or to any entropy taken from it, and weight-sum
   normalization is unaffected.

   `buildMaet` does not currently allow this. Its per-event validation
   requires at least `r` valid values on every attribute, so an event empty
   on one attribute is rejected; only a wholly empty single-attribute
   collection is accepted, as the comment at `buildMaet.m:391` records.
   Relaxing that validation — treating `K_{a,n} = 0` as an event
   contributing no tuple on that attribute rather than as an error — is a
   prerequisite for the gridder, and is a change to the core rather than to
   the new code.

2. **pandas — resolved: Python takes it.** `pandas.DataFrame` against
   MATLAB `table`, `pandas.Categorical` against MATLAB `categorical`, and
   native selection on both sides. It becomes a hard dependency with a
   version floor at the step where the readers emit frames, and
   `read_score` returns a `DataFrame` rather than a dict, which is a
   breaking change on the Python side independent of the schema change.

3. **The look-ahead for a bend sent just after its note-on** (Section 5).

4. **Migration.** `preMaetFromScore` currently takes `'parts'`, `'chords'`,
   `'chordTolerance'`, and a fixed `'attributes'` vocabulary. Under this
   design `'parts'` goes to table selection, `'chords'` becomes the
   aggregating roles, `'chordTolerance'` becomes the grouping tolerance,
   and `'attributes'` becomes the role specification. The name-value
   interface should stay recognizable to anyone using it today.

## 10. Out of scope, with a known extension point

Continuous controller streams as attributes. The extension, if a question
ever needs it, is for the read result to carry the change streams alongside
the table, consulted only by the gridder, so that re-gridding needs no
re-parse. The cost is that the read result becomes a container rather than
a table, that the attached data does not survive every native table
operation, and that nothing keeps the streams in step with the rows under
selection. None of it is needed by anything written so far, and it can be
added without disturbing the table or anything downstream of it.

Also out of scope: slicing at every change point, which makes an event's
weight an artefact of how densely a controller happened to be recorded; and
deriving a grid table from a note table for sources that do have streams,
which the note table cannot support because it has already discarded them.

## 11. Implementation order

Each step leaves the toolbox working and its tests passing.

1. **`readScore` / `read_score` return a table.** MATLAB `table`, pandas
   `DataFrame`; `channel` separated from `voice`; `fermata` logical; `part`
   categorical with the part names as its categories, absorbing
   `partNames`; the source in the table's description. Existing callers
   break here; this is the one breaking step and carries the migration
   note. pandas becomes a dependency at this step.
2. **Pedal, bend, and the resolved streams.** `soundingDuration`, with the
   re-strike and sostenuto rules; bend into `pitch`, with `noteNumber`
   retained and the MPE range handling; volume and expression into
   `weight`. Testable against constructed files without touching anything
   else.
3. **Admit `K_{a,n} = 0` in `buildMaet`.** Per Section 9.1: an event empty
   on an attribute contributes no tuple there rather than raising. Core
   change, independently testable, and a prerequisite for the gridder.
4. **`gridEvents`.** Table to table, metrical and clock grids, both weight
   policies, grid position and note identifier columns. Reproduce Analysis
   1.1's N = 272 on BWV 347 as the acceptance test.
5. **The conversion.** The three roles, the single-aggregating-category
   rule and its warning, the grouping key, and the absence policy.
   Reproduce Analysis 1.2's three encodings as the acceptance test, since
   they exercise all three roles on one table.
6. **The pre-MAET filter.** Attributes and events, with the group-integrity
   refusal.
7. **Additional read columns.** Articulations and the rest of the MusicXML
   vocabulary as categorical columns; MIDI program and track name. Purely
   additive once step 1 is in.
8. **Documentation.** `USER_GUIDE.md`, `ARCHITECTURE.md`, `MIGRATION.md`,
   `CHANGELOG.md`, and the manuscript's toolbox references where the
   encoding names appear.

Steps 1 and 2 gate the rest. Steps 4, 5, and 6 are independent of one
another and may be taken in any order once 3 is in.
