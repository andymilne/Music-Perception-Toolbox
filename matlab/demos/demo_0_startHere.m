%% demo_0_startHere.m
%  Start here: a guide to the demos of the Music Perception Toolbox.
%
%  This script runs no analysis; running it prints this text. It says
%  where to begin, how the demos fit together, and which demos cover
%  each topic. Each demo opens with a header saying what it shows, and
%  most end their sections with pointers to the demos that go further.
%  The User Guide (USER_GUIDE.md, section 15) describes every demo in
%  more detail. Every MATLAB demo has a Python twin in python/demos/,
%  with the same computations and the same printed numbers.
%
%  ===================================================================
%  WHERE TO BEGIN
%  ===================================================================
%
%  1. demo_overview
%       A quick tour of every function family, from a single chord to
%       multi-attribute tensors and the structural measures. Each
%       section ends with a list of the demos that take it further.
%
%  2. The demo for your material:
%       a score (MusicXML or MIDI)       demo_scoreWorkflow
%       audio recordings                 demo_audioAnalysis
%       chords, scales, or tunings       demo_probeTone, demo_edoApprox
%       rhythms                          demo_rhythmTensors
%       a table of experimental trials   demo_batchProcessing
%
%  3. demo_preprocessing
%       The operations that shape a pre-MAET before its density is
%       built (differencing, binding, translation, weighting, and
%       more), and how they combine.
%
%  4. demo_sweptSimilarity
%       Finding a query in a piece: translation, windows, and what each
%       setting finds.
%
%  5. The JMM demos (demos/jmm/)
%       The worked analyses of the toolbox's article in the Journal of
%       Mathematics and Music, each a complete analysis of a real piece.
%
%  ===================================================================
%  HOW THE DEMOS FIT TOGETHER
%  ===================================================================
%
%  Material enters the toolbox in one of four ways:
%
%  * from a score, read into an attribute table (readScore), one row
%    per note and one column per attribute, and optionally sampled on
%    a time grid (gridAttrTable);
%  * from audio, as spectral peaks (audioPeaks);
%  * made by hand to answer a theoretical question: a chord, scale,
%    tuning, or rhythm written as a vector, or many of them as the
%    rows of a matrix;
%  * from an experiment: the stimuli of a study, one row per trial.
%
%  Analyses built on multi-attribute expectation tensors (MAETs) then
%  follow one path, and most demos take up one stage of it:
%
%    attribute table or vectors -> pre-MAET -> preprocessing
%          -> density (MAET) -> measures
%
%  * A pre-MAET is the material chosen for analysis: which attributes,
%    with what kernel widths and flags (preMaetFromAttrTable from an
%    attribute table, packPreMaet from vectors).
%  * Preprocessing shapes the pre-MAET: differencing, binding,
%    translating, weighting, transforming, selecting, and spectral
%    enrichment.
%  * buildMaet turns it into a density, the MAET, and plotMaet draws
%    it.
%  * Measures read the density or compare two: simMaet (similarity),
%    entropyMaet (entropy), evalMaet (the density at chosen points),
%    massMaet (the mass in a region), and the swept forms
%    sweptSimilarity, sweptEntropy, sweptMass, and sweepSimMaet.
%
%  For single chords, scales, or rhythms, and for a matrix of them,
%  simMaet also takes the vectors directly and builds the densities
%  itself.
%
%  Not every function belongs to the MAET framework. Beside it stand
%  measures that take a pitch or rhythm set directly: harmonicity,
%  roughness, and virtual pitches; balance, evenness, and the other
%  circular (Fourier) measures; and coherence, sameness, edges, mean
%  offset, Markov, and n-tuple entropy. Some use expectation tensors
%  internally; several of the circular and structural measures do
%  not use them at all.
%
%  demo_overview walks the MAET path once and tours the other
%  measures. demo_scoreWorkflow follows the path from a score file to
%  a result; demo_batchProcessing starts from a table of trials. The
%  other demos each go deeper into one stage or one measure.
%
%  ===================================================================
%  THE DEMOS BY TOPIC
%  ===================================================================
%
%  A demo appears under every topic it covers, so some appear several
%  times. JMM demos are listed by number here and described in full
%  under "The JMM analyses" below.
%
%  Scores and attribute tables
%    demo_scoreWorkflow     score -> attribute table -> pre-MAET ->
%                           result, end to end
%    demo_scoreGrid         sampling a score on a time grid
%    demo_scoreCategoricals three ways to encode a categorical column
%                           (voice, part, and the like)
%    demo_preMaetIo         showing, exporting, and importing a
%                           pre-MAET (markdown, LaTeX, CSV)
%    JMM 1.1-1.4 (Bach), 2.1-2.3 (Coltrane)
%
%  Building and reading MAETs
%    demo_overview          the four parameters (r, isRel, isPer,
%                           isExch), drawn
%    demo_maetPlots         every combination of the four parameters,
%                           drawn by each plotMaet method
%    demo_softeningEquivalences
%                           softening an equivalence (transposition,
%                           octave, reordering) by pairing attributes
%    demo_helixBlend        a continuum between pitch-class and
%                           pitch-height similarity
%    demo_tempoInvariance   tempo invariance by degree, via kernelCov
%    JMM 1.2, 1.4, 4.1
%
%  Preprocessing pre-MAETs
%    demo_preprocessing     the operations and their compositions
%    demo_repetitionHandling
%                           repeated pitches: excise, prolong, or count
%    demo_scoreCategoricals roles for categorical columns, and
%                           selectPreMaet between them
%    demo_rhythmTensors     circular differencing and binding
%    JMM 1.3, 2.1, 2.2, 3.2, 4.1
%
%  Searching a piece: swept similarity and entropy
%    demo_sweptSimilarity   sweptSimilarity in depth
%    demo_helixBlend        a time-windowed sweep of a motif
%    demo_tempoInvariance   a sweep under a tempo-invariant kernel
%    demo_batchProcessing   sweepSimMaet against translated copies
%    JMM 1.1, 1.3, 2.3, 3.1, 3.2, 3.3
%
%  Pitch collections, scales, and tunings
%    demo_probeTone         probe-tone profiles, recency weighting, and
%                           continuity
%    demo_edoApprox         how well each EDO approximates a JI chord
%    demo_genChainPcs       similarity as a generator is swept
%    demo_triadSpcsGrid     spectral similarity of 12-EDO triads
%    demo_triadConsonance   five consonance measures over triads
%    demo_sigmaSpace        soft sameness, coherence, and n-tuple
%                           entropy of the diatonic scale
%    demo_dftCircularSimulate
%                           balance and evenness under positional
%                           jitter
%
%  Audio, spectra, and consonance
%    demo_audioAnalysis     spectral peaks from audio, then similarity,
%                           harmonicity, roughness, virtual pitches
%    demo_virtualPitches    virtual-pitch salience of chords
%    demo_triadConsonance   harmonicity, spectral entropy, roughness
%    demo_overview          a chord as a spectral density (section 1a)
%    JMM 1.1, 2.3
%
%  Rhythm and time
%    demo_rhythmTensors     the son clave against five other timelines:
%                           similarity, density, and complexity
%    demo_tempoInvariance   logarithmic inter-onset intervals
%    demo_repetitionHandling
%                           repetition and tempo invariance
%    demo_dftCircularSimulate
%                           balance and evenness of rhythms
%    JMM 2.1, 3.1-3.3 (Reich)
%
%  Entropy and structure
%    demo_sigmaSpace        coherence, sameness, and n-tuple entropy
%    demo_rhythmTensors     Rényi-2 entropy as rhythmic complexity
%    demo_dispatchAndKernelControls
%                           the three entropy estimators compared
%    JMM 1.1, 3.1, 3.2
%
%  Experimental data and performance
%    demo_batchProcessing   features for a table of trials, computed
%                           with internal deduplication
%    demo_dispatchAndKernelControls
%                           method dispatch, one-pass sweeps, kernel
%                           truncation and precision, and the defaults
%
%  ===================================================================
%  THE JMM ANALYSES (demos/jmm/)
%  ===================================================================
%
%  Numbered by piece: 1.x Bach, 2.x Coltrane, 3.x Reich, 4.x supplied
%  parses. demos/jmm/README.md maps each to its section of the article
%  or Online Supplement.
%
%    1.1 demo_jmm_1_1_entropy          windowed spectral entropy of
%                                      BWV 347
%    1.2 demo_jmm_1_2_similarity       chord similarity under three
%                                      voicing encodings
%    1.3 demo_jmm_1_3_cadence_nesting  cadences located with nested
%                                      prototypes
%    1.4 demo_jmm_1_4_tonic_tuple_size chord matching as the tuple size
%                                      rises
%    2.1 demo_jmm_2_1_joint            the recurring cell as a joint
%                                      pitch-rhythm object
%    2.2 demo_jmm_2_2_motif            the same cell from pitch alone
%    2.3 demo_jmm_2_3_spectral         the motif located in time and key
%    3.1 demo_jmm_3_1_texture          local entropy tracking the phase
%    3.2 demo_jmm_3_2_diff             joint differencing exposing the
%                                      accelerandi
%    3.3 demo_jmm_3_3_xcorr            the phase as the ridge of a lag
%                                      cross-correlogram
%    4.1 demo_jmm_4_1_parse            a supplied harmonic parse as a
%                                      nested multiset
%
%  The Coltrane demos (2.x) need a MIDI transcription of the solo, and
%  4.1 needs the parse file from its authors' repository; neither is
%  distributed. demos/jmm/README.md says where to place them.
%
%  The Python mirror is demo_0_start_here.py.

help(mfilename)
