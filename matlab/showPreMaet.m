function out = showPreMaet(varargin)
%SHOWPREMAET  Print a pre-MAET as a table, in the layout of Milne (2026).
%
%   SHOWPREMAET(PM) and SHOWPREMAET(PATTR, WATTR, SPECS) print the
%   pre-MAET as a markdown table:
%   one row per attribute and one column per event. An attribute's row is
%   headed by its name and the parameters that determine its density --
%   sigma, the tuple size r, the [rel] and [per] flags, and the period
%   where it is periodic -- and its cells hold the elements from which
%   the admitted tuples are formed. A cell is brace-delimited where the
%   attribute is unordered ([sym] = 1) and parenthesis-delimited where it
%   is ordered; a nested attribute is bracketed level by level, the
%   outermost level outermost. A single element is written bare. Where
%   the weights are not uniform they are written as parenthesized
%   superscripts on their values, 60^(0.6). The markdown rendering is
%   plain ASCII, so that its column widths are the same in MATLAB and in
%   Octave and it survives any terminal encoding; the LaTeX rendering
%   carries the article's own symbols.
%
%   SHOWPREMAET(DENS) takes a density built by buildExpTens, from which
%   every field is recovered.
%
%   STR = SHOWPREMAET(...) also returns the rendered table.
%
%   Name-value arguments:
%     'sigma', 'isRel', 'isPer', 'period'
%                     Kernel parameters, scalar or length A, shown in the
%                     attribute row. Ignored when a density is passed;
%                     'isRel' overrides the specs' rel field.
%     'names'         Attribute names, a string or length-A cell array.
%                     Defaults to the specs' name fields, then to a_1,
%                     a_2, and so on.
%     'format'        'markdown' (default) or 'latex' (a booktabs
%                     tabular in the article's own markup).
%     'maxEvents'     Columns shown before the middle ones are elided
%                     (default 8); [] shows every event.
%     'maxElements'   Elements shown within one cell before the rest are
%                     elided (default 8); [] shows every element.
%     'decimals'      Decimal places for values and weights (default 4);
%                     trailing zeros are dropped.
%     'weights'       'auto' (default) shows the weights where they are
%                     supplied and not uniform; true or false forces it.
%     'title'         A line printed above the markdown table.
%     'caption', 'label'
%                     LaTeX caption and label.
%     'verbose'       Print the table (default true).
%
%   Example:
%     p = {[67 66 64], [5 6 7]};
%     showPreMaet(p, [], [], 'names', {'pitch', 'time'}, ...
%                 'sigma', [0.5 0.25], 'isPer', [true false], ...
%                 'period', [12 0]);
%
%   See also buildExpTens, flatSpecs, differenceEvents, bindEvents.

varargin = internal.expandPreMaet(varargin, 1);
if nargout > 0
    out = localShowPreMaet(varargin{:});
else
    localShowPreMaet(varargin{:});
end
end


function out = localShowPreMaet(pAttr, w, specs, varargin)
if nargin < 2, w = []; end
if nargin < 3, specs = []; end

nvDefaults = struct( ...
    'sigma', [], 'isRel', [], 'isPer', [], 'period', [], ...
    'names', [], 'format', 'markdown', 'maxEvents', 8, ...
    'maxElements', 8, 'decimals', 4, 'weights', 'auto', ...
    'title', [], 'caption', [], 'label', [], 'verbose', true, ...
    'headings', [], 'delimiter', ',');
nv = localParseNV(varargin, nvDefaults);

if ~(ischar(nv.format) || (isstring(nv.format) && isscalar(nv.format))) ...
        || ~any(strcmpi(char(nv.format), {'markdown', 'latex', 'csv'}))
    error('showPreMaet:format', ...
        'format must be ''markdown'', ''latex'' or ''csv''.');
end
isLatex = strcmpi(char(nv.format), 'latex');
isCsv = strcmpi(char(nv.format), 'csv');

[P, W, sp, prm, names] = localUnpack(pAttr, w, specs, nv);
A = numel(P);
if A == 0
    error('showPreMaet:empty', ...
        'A pre-MAET must carry at least one attribute.');
end
N = size(P{1}, 2);
for a = 1:A
    if size(P{a}, 2) ~= N
        error('showPreMaet:events', ...
            ['Attribute %d has %d events but attribute 1 has %d; every ' ...
             'attribute must span the same passage.'], ...
            a, size(P{a}, 2), N);
    end
end

if ischar(nv.weights) || isstring(nv.weights)
    showW = ~isempty(W) && ~localAllUniform(W, P);
else
    showW = logical(nv.weights) && ~isempty(W);
end

if isCsv
    % A file is not a display: it carries the whole pre-MAET, so neither
    % events nor elements are elided.
    nv.maxEvents = [];
    nv.maxElements = [];
end
cols = localEventColumns(N, nv.maxEvents);
if isLatex
    gap = '$\cdots$';
else
    gap = '...';
end

cells = cell(A, numel(cols));
for a = 1:A
    for j = 1:numel(cols)
        if isnan(cols(j))
            cells{a, j} = gap;
            continue;
        end
        if isempty(W)
            wCol = [];
        else
            wCol = W{a}(:, cols(j));
        end
        [node, symLevels] = localCellTree( ...
            P{a}(:, cols(j)), wCol, sp{a}, nv.maxElements);
        cells{a, j} = localRenderNode( ...
            node, symLevels, nv.decimals, showW, isLatex, true);
    end
end

stubs = cell(A, 1);
for a = 1:A
    stubs{a} = localStub(sp{a}, prm, a, isLatex);
end

if isCsv
    % A file records what the pre-MAET holds, so weights are decided per
    % attribute rather than for the table as a whole: an attribute whose
    % weights are all exactly 1 is written bare, since unit weights are
    % what a bare value means, and any other weight is written.
    for a = 1:A
        live = isfinite(P{a});
        if ~isempty(W)
            live = live & isfinite(W{a});
        end
        showWa = ~isempty(W) && any(live(:)) && ...
                 ~all(abs(W{a}(live) - 1) < 1e-12);
        for j = 1:numel(cols)
            if isempty(W)
                wCol = [];
            else
                wCol = W{a}(:, cols(j));
            end
            [node, symLevels] = localCellTree( ...
                P{a}(:, cols(j)), wCol, sp{a}, nv.maxElements);
            cells{a, j} = localRenderNode( ...
                node, symLevels, nv.decimals, showWa, false, true);
        end
    end
    str = localRenderCsv(names, sp, prm, cells, cols, nv);
elseif isLatex
    str = localRenderLatex(names, stubs, cells, cols, N, ...
        nv.caption, nv.label);
else
    str = localRenderMarkdown(names, stubs, cells, cols, N, nv.title);
end

if nv.verbose
    fprintf('%s\n', str);
end
if nargout > 0
    out = str;
end
end


% ===================================================================
%  Input normalisation
% ===================================================================

function [P, W, sp, prm, names] = localUnpack(pAttr, w, specs, nv)
if isstruct(pAttr) && isfield(pAttr, 'pAttr') && isfield(pAttr, 'nAttrs')
    d = pAttr;
    A = double(d.nAttrs);
    P = cell(1, A);
    for a = 1:A
        P{a} = double(d.pAttr{a});
    end
    if isempty(d.w)
        W = [];
    else
        W = cell(1, A);
        for a = 1:A
            W{a} = double(d.w{a});
        end
    end
    sp = cell(1, A);
    for a = 1:A
        if isfield(d, 'nested') && ~isempty(d.nested) ...
                && ~isempty(d.nested{a})
            sp{a} = d.nested{a};
        else
            sp{a} = struct('r', double(d.r(a)), ...
                'rel', logical(d.isRel(a)), 'sym', logical(d.isSym(a)));
        end
    end
    prm = struct('sigma', {num2cell(double(d.sigma(:)'))}, ...
        'isPer', {num2cell(logical(d.isPer(:)'))}, ...
        'period', {num2cell(double(d.period(:)'))});
    names = localNames(nv.names, sp, A);
    return;
end

if ~iscell(pAttr)
    pAttr = {pAttr};
end
A = numel(pAttr);
P = cell(1, A);
for a = 1:A
    P{a} = double(pAttr{a});
    if isvector(P{a}) && size(P{a}, 1) > 1 && size(P{a}, 2) == 1
        % A column vector of one event's elements stays as given.
    elseif isvector(P{a})
        P{a} = reshape(P{a}, 1, []);
    end
end

if isempty(w)
    W = [];
else
    if ~iscell(w), w = {w}; end
    if numel(w) ~= A
        error('showPreMaet:wSize', ...
            'w has %d attributes but pAttr has %d.', numel(w), A);
    end
    W = cell(1, A);
    for a = 1:A
        W{a} = double(w{a});
        if isvector(W{a}) && size(W{a}, 1) == 1
            W{a} = reshape(W{a}, size(P{a}));
        end
    end
end

if isempty(specs)
    sp = cell(1, A);
    for a = 1:A
        sp{a} = struct('r', 1, 'rel', false, 'sym', true);
    end
else
    if ~iscell(specs), specs = {specs}; end
    if numel(specs) ~= A
        error('showPreMaet:specsSize', ...
            'specs has %d attributes but pAttr has %d.', numel(specs), A);
    end
    sp = specs;
end

% A spec may carry the attribute's kernel geometry (Milne 2026,
% Def. 2.6); an explicit argument overrides it, as at build time.
prm = struct( ...
    'sigma', {localBcast(nv.sigma, A, 'sigma')}, ...
    'isPer', {localBcast(nv.isPer, A, 'isPer')}, ...
    'period', {localBcast(nv.period, A, 'period')});
kFields = {'sigma', 'isPer', 'period'};
kAliases = {'sigma', 'is_per', 'period'};
for a = 1:A
    for kf = 1:numel(kFields)
        if ~isempty(prm.(kFields{kf}){a})
            continue;
        end
        if isfield(sp{a}, kFields{kf})
            prm.(kFields{kf}){a} = sp{a}.(kFields{kf});
        elseif isfield(sp{a}, kAliases{kf})
            prm.(kFields{kf}){a} = sp{a}.(kAliases{kf});
        end
    end
end

if ~isempty(nv.isRel)
    relV = localBcast(nv.isRel, A, 'isRel');
    for a = 1:A
        sp{a}.rel = logical(relV{a});
    end
end
names = localNames(nv.names, sp, A);
end


function out = localBcast(v, A, what)
%LOCALBCAST  Per-attribute values, tolerating a non-scalar entry.
%
%   An attribute carrying a kernel covariance (Sec. maet-cov) has a
%   matrix where the others have a width, so the values arrive in a cell
%   rather than a numeric vector. Entries are taken as given.
out = cell(1, A);
if isempty(v)
    return;
end
if iscell(v)
    if numel(v) == 1
        out(:) = v(1);
        return;
    end
    if numel(v) ~= A
        error('showPreMaet:bcast', ...
            '%s must be a scalar or length %d.', what, A);
    end
    out = v(:)';
    return;
end
v = v(:)';
if numel(v) == 1
    v = repmat(v, 1, A);
elseif numel(v) ~= A
    error('showPreMaet:bcast', ...
        '%s must be a scalar or length %d.', what, A);
end
for a = 1:A
    out{a} = v(a);
end
end


function names = localNames(given, sp, A)
names = cell(1, A);
for a = 1:A
    if ~isempty(given)
        if ischar(given) || (isstring(given) && isscalar(given))
            names{a} = char(given);
            continue;
        end
        if numel(given) ~= A
            error('showPreMaet:names', ...
                'names must be a string or length %d.', A);
        end
        names{a} = char(given{a});
        continue;
    end
    if isfield(sp{a}, 'name') && ~isempty(sp{a}.name)
        names{a} = char(sp{a}.name);
    else
        names{a} = sprintf('a_%d', a);
    end
end
end


function v = localLevels(spec, field, default)
if isfield(spec, field) && ~isempty(spec.(field))
    v = spec.(field);
else
    v = default;
end
v = double(v(:)');
end


% ===================================================================
%  Cell model
% ===================================================================

function [node, symLevels] = localCellTree(pCol, wCol, spec, maxElements)
finite = isfinite(pCol);
symLevels = localLevels(spec, 'sym', 1);

if ~isfield(spec, 'tags') || isempty(spec.tags)
    node = localLeaves(find(finite), pCol, wCol, maxElements); %#ok<FNDSB>
    return;
end

T = spec.tags;
if isvector(T)
    % A single grouping column arrives as a vector; it is a column of the
    % tag matrix, not a row of it.
    T = reshape(T, [], 1);
end
if size(T, 1) ~= numel(pCol)
    error('showPreMaet:tags', ...
        'Spec tags have %d rows but the attribute carries %d.', ...
        size(T, 1), numel(pCol));
end
node = localBuildGroups((1:size(T, 1))', T, size(T, 2), ...
    pCol, wCol, finite, maxElements);
end


function node = localBuildGroups(rows, T, col, pCol, wCol, finite, maxEl)
if col < 1
    node = localLeaves(rows(finite(rows)), pCol, wCol, maxEl);
    return;
end
tagVals = T(rows, col);
seen = [];
node = {};
for k = 1:numel(tagVals)
    g = tagVals(k);
    if any(seen == g)
        continue;
    end
    seen(end + 1) = g; %#ok<AGROW>
    sub = rows(tagVals == g);
    child = localBuildGroups(sub, T, col - 1, pCol, wCol, finite, maxEl);
    if ~isempty(child)
        node{end + 1} = child; %#ok<AGROW>
    end
end
end


function leaves = localLeaves(idx, pCol, wCol, maxElements)
idx = idx(:)';
n = numel(idx);
if ~isempty(maxElements) && n > maxElements
    keep = maxElements - 1;
    leaves = cell(1, keep + 1);
    for k = 1:keep
        leaves{k} = localLeaf(idx(k), pCol, wCol);
    end
    leaves{keep + 1} = '...';
    return;
end
leaves = cell(1, n);
for k = 1:n
    leaves{k} = localLeaf(idx(k), pCol, wCol);
end
end


function lf = localLeaf(k, pCol, wCol)
if isempty(wCol)
    lf = [pCol(k), NaN];
else
    lf = [pCol(k), wCol(k)];
end
end


function s = localRenderNode(node, symLevels, decimals, showW, isLatex, top)
d = localDepth(node);
if d < numel(symLevels)
    sym = symLevels(d + 1);
elseif ~isempty(symLevels)
    sym = symLevels(end);
else
    sym = 1;
end
if d == 0
    s = localRenderLeaves(node, decimals, showW, isLatex, sym, top);
    return;
end
sep = ', ';
parts = cell(1, numel(node));
for k = 1:numel(node)
    parts{k} = localRenderNode(node{k}, symLevels, decimals, showW, ...
        isLatex, false);
end
s = localBracket(localJoin(parts, sep), sym, isLatex);
end


function d = localDepth(node)
d = 0;
while ~isempty(node) && iscell(node) && iscell(node{1})
    d = d + 1;
    node = node{1};
end
end


function s = localRenderLeaves(leaves, decimals, showW, isLatex, sym, top)
items = cell(1, numel(leaves));
for k = 1:numel(leaves)
    lf = leaves{k};
    if ischar(lf)
        if isLatex
            items{k} = '\dots';
        else
            items{k} = '...';
        end
        continue;
    end
    s1 = localNum(lf(1), decimals);
    if showW && ~isnan(lf(2))
        ws = localNum(lf(2), decimals);
        if isLatex
            s1 = sprintf('%s^{(%s)}', s1, ws);
        else
            s1 = sprintf('%s^(%s)', s1, ws);
        end
    end
    items{k} = s1;
end
if top && numel(items) == 1
    s = items{1};
    return;
end
s = localBracket(localJoin(items, ', '), sym, isLatex);
end


function s = localBracket(inner, sym, isLatex)
if sym
    if isLatex
        s = ['\{' inner '\}'];
    else
        s = ['{' inner '}'];
    end
else
    s = ['(' inner ')'];
end
end


function s = localNum(x, decimals)
if isnan(x)
    s = '-';
    return;
end
if isinf(x)
    if x > 0
        s = '\infty';
    else
        s = '-\infty';
    end
    return;
end
s = sprintf('%.*f', decimals, x);
if any(s == '.')
    s = regexprep(s, '0+$', '');
    s = regexprep(s, '\.$', '');
end
if isempty(s) || strcmp(s, '-') || strcmp(s, '-0') || strcmp(s, '0')
    % A value that rounds to all zeros is shown in scientific notation
    % rather than as an exact zero, since a table is read for which
    % entries vanish and a rounded-away tail is not one of them. Below
    % the dirt floor, six decades finer than the requested precision, the
    % value is floating-point residue and is shown as the zero it is
    % meant to be.
    if abs(x) < 0.5 * 10^-(decimals + 6)
        s = '0';
    else
        s = sprintf('%.*e', max(decimals - 2, 1), x);
    end
end
end


function s = localParamNum(x)
%LOCALPARAMNUM  A parameter as it appears in the attribute row.
%
%   A kernel covariance is named by its shape rather than printed: the
%   row states the density's settings, and a matrix belongs in the text
%   that discusses it.
if isempty(x)
    s = '';
    return;
end
if ~isscalar(x) && ndims(x) >= 2 && min(size(x)) > 1 %#ok<ISMAT>
    s = sprintf('%dx%d covariance', size(x, 1), size(x, 2));
    return;
end
% NA is shown as such rather than as a number or an omission: a
% preprocessing step could not carry this parameter forward, and the
% table is where the user should see that before the build refuses it.
if isnan(double(x(1)))
    s = 'NA';
else
    s = sprintf('%.6g', double(x(1)));
end
end


% ===================================================================
%  Attribute stub
% ===================================================================

function stub = localStub(spec, prm, a, isLatex)
r = localLevels(spec, 'r', 1);
rel = localLevels(spec, 'rel', 0);
sigma = prm.sigma{a};
isPer = prm.isPer{a};
period = prm.period{a};

if isLatex
    sig = '\sigma';
    relT = '{[\mathrm{rel}]}';
    perT = '{[\mathrm{per}]}';
else
    sig = 'sigma';
    relT = '[rel]';
    perT = '[per]';
end
sep = ', ';

bits = {};
if ~isempty(sigma)
    bits{end + 1} = sprintf('%s = %s', sig, localParamNum(sigma)); %#ok<AGROW>
end
bits{end + 1} = ['r = ' localTuple(r)];

if isempty(isPer)
    perV = 0;
else
    perV = double(logical(isPer));
end
if numel(rel) == 1 && rel(1) == 0 && perV == 0
    % The article collapses the two flags where both are the scalar 0.
    bits{end + 1} = sprintf('%s, %s = 0', relT, perT);
else
    bits{end + 1} = [relT ' = ' localTuple(rel)];
    bits{end + 1} = sprintf('%s = %d', perT, perV);
    if perV && ~isempty(period) && period ~= 0
        bits{end + 1} = ['P = ' localParamNum(period)];
    end
end
stub = localJoin(bits, sep);
end


function s = localTuple(v)
if numel(v) == 1
    s = sprintf('%d', round(v));
    return;
end
parts = cell(1, numel(v));
for k = 1:numel(v)
    parts{k} = sprintf('%d', round(v(k)));
end
s = ['(' localJoin(parts, ', ') ')'];
end


% ===================================================================
%  Column selection and rendering
% ===================================================================

function cols = localEventColumns(N, maxEvents)
if isempty(maxEvents) || N <= maxEvents
    cols = 1:N;
    return;
end
head = maxEvents - 2;
cols = [1:head, NaN, N];
end


function str = localRenderMarkdown(names, stubs, cells, cols, N, ttl)
nCol = numel(cols);
A = numel(names);
headers = cell(1, nCol);
for j = 1:nCol
    if isnan(cols(j))
        headers{j} = '...';
    else
        headers{j} = sprintf('n = %d', cols(j));
    end
end

stubCol = cell(1, A);
for a = 1:A
    stubCol{a} = sprintf('%s: %s', names{a}, stubs{a});
end

leftW = max(cellfun(@numel, [stubCol, {'attribute'}]));
colW = zeros(1, nCol);
for j = 1:nCol
    widths = numel(headers{j});
    for a = 1:A
        widths = max(widths, numel(cells{a, j}));
    end
    colW(j) = widths;
end

lines = {};
if ~isempty(ttl)
    lines{end + 1} = char(ttl); %#ok<AGROW>
    lines{end + 1} = ''; %#ok<AGROW>
end
lines{end + 1} = localMdRow('attribute', headers, leftW, colW); %#ok<AGROW>

seps = cell(1, nCol);
for j = 1:nCol
    seps{j} = [':' repmat('-', 1, colW(j)) ':'];
end
lines{end + 1} = ['|:' repmat('-', 1, leftW) '-|' ...
    localJoin(seps, '|') '|']; %#ok<AGROW>

for a = 1:A
    lines{end + 1} = localMdRow(stubCol{a}, cells(a, :), leftW, colW); %#ok<AGROW>
end
str = localJoin(lines, sprintf('\n'));
end


function line = localMdRow(left, fields, leftW, colW)
parts = cell(1, numel(fields));
for j = 1:numel(fields)
    parts{j} = localCentre(fields{j}, colW(j));
end
line = ['| ' localPadRight(left, leftW) ' | ' ...
    localJoin(parts, ' | ') ' |'];
end


function s = localCentre(s0, width)
pad = width - numel(s0);
if pad <= 0
    s = s0;
    return;
end
lhs = floor(pad / 2);
s = [repmat(' ', 1, lhs) s0 repmat(' ', 1, pad - lhs)];
end


function s = localPadRight(s0, width)
if numel(s0) >= width
    s = s0;
else
    s = [s0 repmat(' ', 1, width - numel(s0))];
end
end


function str = localRenderLatex(names, stubs, cells, cols, N, caption, label)
nCol = numel(cols);
A = numel(names);
headers = cell(1, nCol);
for j = 1:nCol
    if isnan(cols(j))
        headers{j} = '$\cdots$';
    else
        headers{j} = sprintf('$n = %d$', cols(j));
    end
end

lines = {'\begin{table}[]', '\centering', '\footnotesize'};
if ~isempty(caption)
    lines{end + 1} = ['\caption{' char(caption) '}']; %#ok<AGROW>
end
if ~isempty(label)
    lines{end + 1} = ['\label{' char(label) '}']; %#ok<AGROW>
end
lines{end + 1} = '\smallskip'; %#ok<AGROW>
lines{end + 1} = ['\begin{tabular}{@{}' repmat('c', 1, nCol + 1) '@{}}']; %#ok<AGROW>
lines{end + 1} = '\toprule'; %#ok<AGROW>
lines{end + 1} = ['attribute & ' localJoin(headers, ' & ') ' \\']; %#ok<AGROW>
lines{end + 1} = '\midrule'; %#ok<AGROW>
for a = 1:A
    stub = ['$\begin{array}{@{}c@{}} \text{' names{a} '} \\ ' ...
        stubs{a} ' \end{array}$'];
    row = cell(1, nCol);
    for j = 1:nCol
        c = cells{a, j};
        if ~isempty(c) && c(1) == '$'
            row{j} = c;
        else
            row{j} = ['$' c '$'];
        end
    end
    lines{end + 1} = [stub ' & ' localJoin(row, ' & ') ' \\']; %#ok<AGROW>
    if a ~= A
        lines{end + 1} = '\midrule'; %#ok<AGROW>
    end
end
lines{end + 1} = '\bottomrule'; %#ok<AGROW>
lines{end + 1} = '\end{tabular}'; %#ok<AGROW>
lines{end + 1} = '\end{table}'; %#ok<AGROW>
str = localJoin(lines, sprintf('\n'));
end


function tf = localAllUniform(W, P)
tf = true;
ref = [];
for a = 1:numel(W)
    live = isfinite(P{a}) & isfinite(W{a});
    v = W{a}(live);
    if isempty(v)
        continue;
    end
    if isempty(ref)
        ref = v(1);
    end
    if any(abs(v - ref) > 1e-12 * max(1, abs(ref)))
        tf = false;
        return;
    end
end
end


function s = localJoin(parts, sep)
%LOCALJOIN  Concatenate with a literal separator.
%
%   strjoin expands escape sequences in its delimiter, so a separator is
%   not guaranteed to survive it verbatim. This joins the parts as given.
if isempty(parts)
    s = '';
    return;
end
s = parts{1};
for k = 2:numel(parts)
    s = [s sep parts{k}]; %#ok<AGROW>
end
end


function nv = localParseNV(args, defaults)
nv = defaults;
if mod(numel(args), 2) ~= 0
    error('showPreMaet:nvPairs', ...
        'Name-value arguments must come in pairs.');
end
fn = fieldnames(defaults);
for k = 1:2:numel(args)
    name = args{k};
    if ~(ischar(name) || (isstring(name) && isscalar(name)))
        error('showPreMaet:nvName', ...
            'Argument %d should be a name.', k);
    end
    hit = find(strcmpi(char(name), fn), 1);
    if isempty(hit)
        error('showPreMaet:nvUnknown', ...
            'Unknown argument ''%s''.', char(name));
    end
    nv.(fn{hit}) = args{k + 1};
end
end


function str = localRenderCsv(names, sp, prm, cells, cols, nv)
%LOCALRENDERCSV  The pre-MAET as CSV, in the format readPreMaet reads.
%
%   One row per attribute: the fixed parameter columns name, sigma, r,
%   rel, per, P, sym, then one column per event carrying the same cell
%   grammar the markdown and LaTeX renderings use.
A = numel(names);
nCol = numel(cols);
headings = cell(1, nCol);
for j = 1:nCol
    if isempty(nv.headings)
        headings{j} = sprintf('n = %d', cols(j));
    else
        headings{j} = nv.headings{j};
    end
end
lines = {localJoinCsv([{'name', 'sigma', 'r', 'rel', 'per', 'P', 'sym'}, ...
                       headings], nv.delimiter)};
for a = 1:A
    row = {localCsvName(names{a}), ...
           localFmtParam(prm.sigma{a}), ...
           localFmtParam(localSpecLevels(sp{a}, 'r', 1)), ...
           localFmtParam(localSpecLevels(sp{a}, 'rel', 0)), ...
           localFmtParam(localCsvFlag(prm.isPer{a})), ...
           localFmtParam(prm.period{a}), ...
           localFmtParam(localSpecLevels(sp{a}, 'sym', 1))};
    for j = 1:nCol
        row{end+1} = cells{a, j}; %#ok<AGROW>
    end
    lines{end+1} = localJoinCsv(row, nv.delimiter); %#ok<AGROW>
end
str = lines{1};
for i = 2:numel(lines)
    str = [str sprintf('\n') lines{i}]; %#ok<AGROW>
end
str = [str sprintf('\n')];
end


function s = localCsvName(nm)
if isempty(nm); s = ''; else; s = char(nm); end
end


function v = localSpecLevels(spec, key, dflt)
if isfield(spec, key) && ~isempty(spec.(key))
    v = double(spec.(key)(:)');
else
    v = dflt;
end
end


function v = localCsvFlag(x)
if isempty(x)
    v = [];
elseif isnan(double(x(1)))
    v = NaN;
else
    v = double(logical(x(1)));
end
end


function s = localFmtParam(v)
%LOCALFMTPARAM  A parameter cell: '' absent, NA, a value, or a tuple.
%
%   A kernel covariance is written as the three scalars that generate it;
%   one outside that family has no such scalars and is refused.
if isempty(v)
    s = '';
    return;
end
if ~isscalar(v) && min(size(v)) > 1
    prm = internal.preMaetCovParams(v);
    if isempty(prm)
        error('showPreMaet:covToCsv', ...
            ['This kernel covariance cannot be written to CSV. The ' ...
             'format carries a covariance as the three scalars that ' ...
             'generate it (sdPosition, sdInterval, sdShift), and this ' ...
             'matrix is not of that family -- so there are no such ' ...
             'scalars to write. Set it on the spec after reading ' ...
             'instead.']);
    end
    % The written spelling is the same in both languages, so an exported
    % file is byte-identical whichever wrote it; either spelling is read.
    s = sprintf('cov(sd_position=%g, sd_interval=%g, sd_shift=%g)', ...
                prm(1), prm(2), prm(3));
    return;
end
if isscalar(v)
    if isnan(double(v)); s = 'NA'; else; s = sprintf('%g', double(v)); end
    return;
end
parts = cell(1, numel(v));
for k = 1:numel(v)
    if isnan(double(v(k)))
        parts{k} = 'NA';
    else
        parts{k} = sprintf('%g', double(v(k)));
    end
end
s = ['(' localJoin(parts, ', ') ')'];
end


function line = localJoinCsv(fields, delimiter)
%LOCALJOINCSV  Join fields, quoting any that hold a delimiter or a quote.
parts = cell(1, numel(fields));
for k = 1:numel(fields)
    f = char(fields{k});
    if any(f == delimiter) || any(f == '"') || any(f == sprintf('\n'))
        f = ['"' strrep(f, '"', '""') '"'];
    end
    parts{k} = f;
end
line = localJoin(parts, delimiter);
end
