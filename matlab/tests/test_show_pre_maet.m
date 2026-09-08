%% test_show_pre_maet.m — showPreMaet
%
%  Tests for showPreMaet, the pre-MAET table renderer.
%
%  The rendering is a fixed string, so most assertions here are on exact
%  output: the markdown table is cross-language (its column widths must
%  be identical in MATLAB and Python, which is why it is plain ASCII),
%  and the LaTeX table is pasted into a manuscript. Anything that changes
%  these strings changes a published artefact, so the tests state them in
%  full rather than probing for substrings.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

pFrag = {[69 69 69 71 67 66 64], [1 2 3 4 5 6 7]};
metre = [1 0.5 0.75 0.5 1 0.5 0.75];
wFrag = {metre, metre};
KW = {'names', {'pitch', 'time'}, 'sigma', [0.5 0.25], ...
      'isPer', [true false], 'period', [12 0], 'verbose', false};

% ---- Markdown ----

out = showPreMaet(pFrag, wFrag, [], KW{:});
expected = [ ...
'| attribute                                               | n = 1  |  n = 2   |   n = 3   |  n = 4   | n = 5  |  n = 6   |   n = 7   |' sprintf('\n') ...
'|:--------------------------------------------------------|:------:|:--------:|:---------:|:--------:|:------:|:--------:|:---------:|' sprintf('\n') ...
'| pitch: sigma = 0.5, r = 1, [rel] = 0, [per] = 1, P = 12 | 69^(1) | 69^(0.5) | 69^(0.75) | 71^(0.5) | 67^(1) | 66^(0.5) | 64^(0.75) |' sprintf('\n') ...
'| time: sigma = 0.25, r = 1, [rel], [per] = 0             | 1^(1)  | 2^(0.5)  | 3^(0.75)  | 4^(0.5)  | 5^(1)  | 6^(0.5)  | 7^(0.75)  |'];
results{end+1,1} = 'showPreMaet: markdown table'; %#ok<SAGROW>
results{end,2}   = strcmp(out, expected);

% The padding must survive into Python, and into Octave, whose char is
% bytewise: every row is the same width.
widthsEqual = true;
for kw = {{}, {'maxEvents', 4}, {'maxElements', 3}}
    o = showPreMaet(pFrag, wFrag, [], KW{:}, kw{1}{:});
    lines = strsplit(o, sprintf('\n'));
    widthsEqual = widthsEqual && numel(unique(cellfun(@numel, lines))) == 1;
end
results{end+1,1} = 'showPreMaet: rows are equal width'; %#ok<SAGROW>
results{end,2}   = widthsEqual;

% A multi-byte glyph would count as several characters in Octave and one
% in MATLAB, so the column widths would diverge.
o = showPreMaet(pFrag, wFrag, [], KW{:}, 'maxEvents', 4);
results{end+1,1} = 'showPreMaet: markdown is ASCII'; %#ok<SAGROW>
results{end,2}   = all(double(o) < 128);

o = showPreMaet(pFrag, [], [], KW{:});
results{end+1,1} = 'showPreMaet: uniform weights not shown'; %#ok<SAGROW>
results{end,2}   = isempty(strfind(o, '^(')); %#ok<STREMP>

o = showPreMaet(pFrag, {ones(1,7), ones(1,7)}, [], KW{:}, 'weights', true);
o2 = showPreMaet(pFrag, wFrag, [], KW{:}, 'weights', false);
results{end+1,1} = 'showPreMaet: weights forced and suppressed'; %#ok<SAGROW>
results{end,2}   = ~isempty(strfind(o, '^(1)')) && isempty(strfind(o2, '^(')); %#ok<STREMP>

o = showPreMaet(pFrag, wFrag, [], KW{:}, 'maxEvents', 4);
results{end+1,1} = 'showPreMaet: event elision'; %#ok<SAGROW>
results{end,2}   = ~isempty(strfind(o, 'n = 1')) && ...
                   ~isempty(strfind(o, 'n = 7')) && ...
                   isempty(strfind(o, 'n = 4')) && ...
                   ~isempty(strfind(o, ' ... ')); %#ok<STREMP>

o = showPreMaet({(60:71)'}, [], [], 'names', {'pitch'}, 'sigma', 0.1, ...
                'maxElements', 5, 'verbose', false);
results{end+1,1} = 'showPreMaet: element elision'; %#ok<SAGROW>
results{end,2}   = ~isempty(strfind(o, '{60, 61, 62, 63, ...}')); %#ok<STREMP>

% ---- Cells ----

P = [36; 55; 60; 64];
oSym = showPreMaet({P}, [], {struct('r',2,'rel',false,'sym',true)}, ...
                   'names', {'pitch'}, 'sigma', 0.15, 'verbose', false);
oOrd = showPreMaet({P}, [], {struct('r',2,'rel',false,'sym',false)}, ...
                   'names', {'pitch'}, 'sigma', 0.15, 'verbose', false);
results{end+1,1} = 'showPreMaet: unordered braces, ordered parens'; %#ok<SAGROW>
results{end,2}   = ~isempty(strfind(oSym, '{36, 55, 60, 64}')) && ...
                   ~isempty(strfind(oOrd, '(36, 55, 60, 64)')); %#ok<STREMP>

o = showPreMaet({[69 66 64]}, [], [], 'names', {'pitch'}, ...
                'sigma', 0.5, 'verbose', false);
results{end+1,1} = 'showPreMaet: single element bare at top level'; %#ok<SAGROW>
results{end,2}   = isempty(strfind(o, '{69}')); %#ok<STREMP>

% The outermost level takes the outermost bracket, as the article writes
% it: an ordered run of unordered chords.
Pn = [36 43; 55 55; 60 59; 64 62];
[pb, wb, sb] = unpackPreMaet(bindEvents({Pn}, [], 2));
o = showPreMaet(pb, wb, sb, 'names', {'pitch'}, 'sigma', 0.15, ...
                'isPer', true, 'period', 12, 'verbose', false);
results{end+1,1} = 'showPreMaet: nested brackets outermost first'; %#ok<SAGROW>
results{end,2}   = ~isempty(strfind(o, ...
    '({36, 55, 60, 64}, {43, 55, 59, 62})')); %#ok<STREMP>

% An inner level of a nest stays bracketed, so the level is visible even
% where it holds a single value.
[pb1, wb1, sb1] = unpackPreMaet(bindEvents({[69 66 64]}, [], 2));
o = showPreMaet(pb1, wb1, sb1, 'names', {'pitch'}, 'sigma', 0.5, ...
                'verbose', false);
results{end+1,1} = 'showPreMaet: nested keeps brackets at one element'; %#ok<SAGROW>
results{end,2}   = ~isempty(strfind(o, '({69}, {66})')); %#ok<STREMP>

o = showPreMaet({[60 60; 64 64; 67 NaN]}, [], ...
                {struct('r',1,'rel',false,'sym',true)}, ...
                'names', {'pitch'}, 'sigma', 0.5, 'verbose', false);
results{end+1,1} = 'showPreMaet: NaN is absent, not an element'; %#ok<SAGROW>
results{end,2}   = ~isempty(strfind(o, '{60, 64, 67}')) && ...
                   ~isempty(strfind(o, '{60, 64}')); %#ok<STREMP>

% ---- Numbers ----

% A table is read for which entries vanish; a rounded-away tail is not
% one of them.
o = showPreMaet({[1 2]}, {[1 3.7e-6]}, [], 'names', {'x'}, ...
                'sigma', 1, 'decimals', 4, 'verbose', false);
results{end+1,1} = 'showPreMaet: rounded-away value not shown as zero'; %#ok<SAGROW>
results{end,2}   = isempty(strfind(o, '^(0)')) && ...
                   ~isempty(strfind(o, 'e-06')); %#ok<STREMP>

% Below the dirt floor the value is residue, not a small number: a
% simplex vertex coordinate must not read -1.96e-17.
o = showPreMaet({[0 -1.96e-17]}, [], [], 'names', {'x'}, 'sigma', 1, ...
                'verbose', false);
results{end+1,1} = 'showPreMaet: floating-point residue is zero'; %#ok<SAGROW>
results{end,2}   = isempty(strfind(o, 'e-17')); %#ok<STREMP>

% ---- Attribute row ----

o = showPreMaet({1}, [], [], 'names', {'x'}, 'sigma', 1, 'verbose', false);
results{end+1,1} = 'showPreMaet: flags collapse when both scalar zero'; %#ok<SAGROW>
results{end,2}   = ~isempty(strfind(o, '[rel], [per] = 0')); %#ok<STREMP>

o = showPreMaet(pFrag, wFrag, [], KW{:});
results{end+1,1} = 'showPreMaet: period shown only when periodic'; %#ok<SAGROW>
results{end,2}   = ~isempty(strfind(o, '[per] = 1, P = 12')) && ...
                   isempty(strfind(o, 'P = 0')); %#ok<STREMP>

% An attribute may carry a covariance where the others carry a width; the
% row states its shape rather than printing a matrix.
[pb3, wb3, sb3] = unpackPreMaet(bindEvents({[1 2 3 4]}, [], 3));
o = showPreMaet(pb3, wb3, sb3, 'names', {'trigram'}, ...
                'sigma', {eye(3) * 0.04}, 'verbose', false);
results{end+1,1} = 'showPreMaet: kernel covariance named by shape'; %#ok<SAGROW>
results{end,2}   = ~isempty(strfind(o, 'sigma = 3x3 covariance')); %#ok<STREMP>

o = showPreMaet({1, 2}, [], ...
                {struct('r',1,'rel',false,'sym',true,'name','pitch'), ...
                 struct('r',1,'rel',false,'sym',true)}, ...
                'sigma', 1, 'verbose', false);
results{end+1,1} = 'showPreMaet: names default to specs then index'; %#ok<SAGROW>
results{end,2}   = ~isempty(strfind(o, '| pitch: ')) && ...
                   ~isempty(strfind(o, '| a_2: ')); %#ok<STREMP>

% ---- Density input ----

% A built density carries every field the raw triple supplies, so the two
% inputs must render the same table.
dens = buildExpTens(pFrag, wFrag, [0.5 0.25], [1 1], [false false], ...
                    [true false], [12 0], 'verbose', false);
oDens = showPreMaet(dens, [], [], 'names', {'pitch', 'time'}, ...
                    'verbose', false);
results{end+1,1} = 'showPreMaet: density and raw agree'; %#ok<SAGROW>
results{end,2}   = strcmp(oDens, showPreMaet(pFrag, wFrag, [], KW{:}));

% ---- LaTeX ----

o = showPreMaet({P}, [], {struct('r',2,'rel',false,'sym',true)}, ...
                'names', {'pitch'}, 'sigma', 0.15, 'isPer', true, ...
                'period', 12, 'format', 'latex', 'caption', 'A caption.', ...
                'label', 'tab:x', 'verbose', false);
expectedTex = [ ...
'\begin{table}[]' sprintf('\n') ...
'\centering' sprintf('\n') ...
'\footnotesize' sprintf('\n') ...
'\caption{A caption.}' sprintf('\n') ...
'\label{tab:x}' sprintf('\n') ...
'\smallskip' sprintf('\n') ...
'\begin{tabular}{@{}cc@{}}' sprintf('\n') ...
'\toprule' sprintf('\n') ...
'attribute & $n = 1$ \\' sprintf('\n') ...
'\midrule' sprintf('\n') ...
'$\begin{array}{@{}c@{}} \text{pitch} \\ \sigma = 0.15, r = 2, {[\mathrm{rel}]} = 0, {[\mathrm{per}]} = 1, P = 12 \end{array}$ & $\{36, 55, 60, 64\}$ \\' sprintf('\n') ...
'\bottomrule' sprintf('\n') ...
'\end{tabular}' sprintf('\n') ...
'\end{table}'];
results{end+1,1} = 'showPreMaet: LaTeX table'; %#ok<SAGROW>
results{end,2}   = strcmp(o, expectedTex);

o = showPreMaet(pFrag, wFrag, [], KW{:}, 'format', 'latex');
results{end+1,1} = 'showPreMaet: LaTeX has no thin space after commas'; %#ok<SAGROW>
results{end,2}   = isempty(strfind(o, [',' char(92) ' '])); %#ok<STREMP>

o = showPreMaet(pFrag, wFrag, [], KW{:}, 'format', 'latex', 'maxEvents', 4);
results{end+1,1} = 'showPreMaet: LaTeX elision, caption optional'; %#ok<SAGROW>
results{end,2}   = ~isempty(strfind(o, '$\cdots$')) && ...
                   isempty(strfind(o, '\caption')); %#ok<STREMP>

% ---- Errors ----

threw = false;
try
    showPreMaet(pFrag, [], [], 'format', 'html', 'verbose', false);
catch
    threw = true;
end
results{end+1,1} = 'showPreMaet: bad format errors'; %#ok<SAGROW>
results{end,2}   = threw;

threw = false;
try
    showPreMaet({[1 2], 1}, [], [], 'sigma', 1, 'verbose', false);
catch
    threw = true;
end
results{end+1,1} = 'showPreMaet: ragged passage errors'; %#ok<SAGROW>
results{end,2}   = threw;

threw = false;
try
    showPreMaet(pFrag, {ones(1,7)}, [], 'verbose', false);
catch
    threw = true;
end
results{end+1,1} = 'showPreMaet: weight attribute count errors'; %#ok<SAGROW>
results{end,2}   = threw;


if standalone
    nPass = 0; nFail = 0;
    for i = 1:size(results, 1)
        if results{i,2}
            nPass = nPass + 1;
            fprintf('  PASS  %s\n', results{i,1});
        else
            nFail = nFail + 1;
            fprintf('  FAIL  %s\n', results{i,1});
        end
    end
    fprintf('\n=== test_show_pre_maet: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_show_pre_maet:failed', '%d test(s) failed.', nFail);
    end
end
