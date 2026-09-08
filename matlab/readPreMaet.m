function pm = readPreMaet(source, varargin)
%READPREMAET  Read a pre-MAET from a CSV file.
%
%   PM = READPREMAET(SOURCE) reads a pre-MAET laid out as a table: one row
%   per attribute, its parameters at the left and one column per event (Milne 2026, Def. 2.6). That is also a spreadsheet,
%   so an analysis can be written in Excel or Numbers, exported as CSV,
%   and read straight in. SOURCE is a path or the CSV text itself.
%
%   The cells use the notation of the article and of showPreMaet: braces
%   for an unordered multiset, parentheses for an ordered one, brackets
%   within brackets for a nested attribute, and 60^(0.6) for a weighted
%   value. writePreMaet is the inverse, so a pre-MAET survives a round
%   trip through a spreadsheet unchanged.
%
%   The header is fixed: name, sigma, r, rel, per, P, sym followed by one
%   column per event, whose own headings are free text. r, rel and sym
%   take a parenthesised tuple on a nested attribute, innermost level
%   first, as the article writes them. An empty parameter cell is absent
%   and NA is NA. A kernel covariance is written as the three scalars
%   that generate it, cov(sdPosition=..., sdInterval=..., sdShift=...);
%   the row's own r gives the order.
%
%   Name-value arguments:
%     'delimiter'  Field separator (default ','); use sprintf('\t') for a
%                  tab-separated export.
%
%   Returns the pre-MAET: its wAttr is [] where no cell
%   carried a weight, and each spec holds r, rel, sym, name and, where the file
%   gives them, sigma, isPer and period. A nested attribute also carries
%   its tags, reconstructed from the bracket structure of its cells.
%
%   See also WRITEPREMAET, SHOWPREMAET, BUILDEXPTENS.

nv = struct('delimiter', ',');
if mod(numel(varargin), 2) ~= 0
    error('readPreMaet:nvPairs', 'Name-value arguments must come in pairs.');
end
for k = 1:2:numel(varargin)
    nv.(lower(char(varargin{k}))) = varargin{k + 1};
end

rows = localRows(source, nv.delimiter);
if isempty(rows)
    error('readPreMaet:empty', 'The pre-MAET file is empty.');
end
PARAMS = {'name', 'sigma', 'r', 'rel', 'per', 'p', 'sym'};
header = rows{1};
for k = 1:numel(PARAMS)
    if numel(header) < k || ~strcmpi(strtrim(header{k}), PARAMS{k})
        error('readPreMaet:header', ...
            ['The header must begin name, sigma, r, rel, per, P, sym ' ...
             'followed by one column per event.']);
    end
end
nPar = numel(PARAMS);

body = {};
for i = 2:numel(rows)
    if any(~cellfun(@(c) isempty(strtrim(c)), rows{i}))
        body{end+1} = rows{i}; %#ok<AGROW>
    end
end
if isempty(body)
    error('readPreMaet:noAttributes', ...
        'The pre-MAET file has a header but no attributes.');
end
N = max(cellfun(@numel, body)) - nPar;
if N < 1
    error('readPreMaet:noEvents', 'The pre-MAET file has no event columns.');
end

A = numel(body);
pAttr = cell(1, A); wCell = cell(1, A); specs = cell(1, A);
anyW = false;
for a = 1:A
    row = body{a};
    row(end+1:nPar + N) = {''};
    name = strtrim(row{1});
    nodes = cell(1, N); symSeen = [];
    for n = 1:N
        [nodes{n}, sym] = internal.preMaetParseCell(row{nPar + n});
        if isempty(symSeen) && ~isempty(sym); symSeen = sym; end
    end
    depth = 0;
    for n = 1:N
        depth = max(depth, internal.preMaetDepth(nodes{n}));
    end
    [P, W, tags] = internal.preMaetStack(nodes, depth, a, name);
    spec = struct();
    if ~isempty(name); spec.name = name; end
    spec.r = localLevels(internal.preMaetParam(row{3}), depth, 1);
    spec.rel = localLevels(internal.preMaetParam(row{4}), depth, 0);
    symCol = internal.preMaetParam(row{7});
    if isempty(symCol) && ~isempty(symSeen)
        symCol = double(symSeen);
    end
    spec.sym = localLevels(symCol, depth, 1);
    if depth > 0; spec.tags = tags; end

    cov = internal.preMaetParseCov(row{2});
    if ~isempty(cov)
        rOuter = spec.r(end);
        spec.sigma = intervalKernelCov(rOuter, ...
            'sdPosition', cov(1), 'sdInterval', cov(2), 'sdShift', cov(3));
    else
        sig = internal.preMaetParam(row{2});
        if ~isempty(sig); spec.sigma = sig; end
    end
    per = internal.preMaetParam(row{5});
    if ~isempty(per)
        if isnan(per); spec.isPer = NaN; else; spec.isPer = logical(per); end
    end
    period = internal.preMaetParam(row{6});
    if ~isempty(period); spec.period = period; end

    pAttr{a} = P; wCell{a} = W; specs{a} = spec;
    anyW = anyW || ~isempty(W);
end

if anyW
    w = cell(1, A);
    for a = 1:A
        if isempty(wCell{a})
            w{a} = ones(size(pAttr{a}));
        else
            w{a} = wCell{a};
        end
        w{a}(isnan(pAttr{a})) = NaN;
    end
else
    w = [];
end

pm = preMaet(pAttr, w, specs);
end


function v = localLevels(val, depth, dflt)
%LOCALLEVELS  A per-level row for a nested attribute, a scalar for a flat one.
if isempty(val)
    if depth > 0
        v = repmat(dflt, 1, depth + 1);
    else
        v = dflt;
    end
    return;
end
v = double(val(:)');
if depth == 0 && numel(v) == 1
    v = v(1);
end
end


function rows = localRows(source, delimiter)
%LOCALROWS  CSV rows as a cell of cell-of-char, honouring quoted fields.
txt = char(source);
if isempty(strfind(txt, sprintf('\n'))) && ...
        isempty(strfind(txt, sprintf('\r'))) %#ok<STREMP>
    fid = fopen(txt, 'r');
    if fid < 0
        error('readPreMaet:open', 'Cannot open %s.', txt);
    end
    txt = fread(fid, '*char')';
    fclose(fid);
end
txt = strrep(txt, sprintf('\r\n'), sprintf('\n'));
lines = strsplit(txt, sprintf('\n'));
rows = {};
for i = 1:numel(lines)
    if isempty(strtrim(lines{i})); continue; end
    rows{end+1} = localSplitCsv(lines{i}, delimiter); %#ok<AGROW>
end
end


function out = localSplitCsv(line, delimiter)
%LOCALSPLITCSV  One CSV line into fields, respecting double quotes.
out = {}; cur = ''; inQ = false; i = 1;
while i <= numel(line)
    ch = line(i);
    if inQ
        if ch == '"'
            if i < numel(line) && line(i + 1) == '"'
                cur(end+1) = '"'; %#ok<AGROW>
                i = i + 1;
            else
                inQ = false;
            end
        else
            cur(end+1) = ch; %#ok<AGROW>
        end
    elseif ch == '"'
        inQ = true;
    elseif ch == delimiter
        out{end+1} = cur; %#ok<AGROW>
        cur = '';
    else
        cur(end+1) = ch; %#ok<AGROW>
    end
    i = i + 1;
end
out{end+1} = cur;
end
