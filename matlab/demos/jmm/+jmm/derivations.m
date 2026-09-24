function t = derivations(tunes, jsonPath)
%DERIVATIONS  The rule-labelled derivations as a table, one row per position.
%
%   t = jmm.derivations()
%   t = jmm.derivations(tunes)
%   t = jmm.derivations(tunes, jsonPath)
%
%   Columns: tune, chord (the surface chord's index within its tune,
%   from 0), level (the position's depth, the root at 1), and label (the
%   rule applied there). Reading a derivation as a table of positions is
%   what lets the demo bind them: the positions of one chord are
%   consecutive rows sharing a chord index.
%
%   tunes selects by name (the corpus prefixes them '(Valid)'); the
%   default reads all 150.
%
%   The derivations are the parses of Ren, Rammos, and Rohrmeier (2024)
%   over the Jazz Harmony Treebank. They are not part of the toolbox
%   distribution: download ParseTrees.json and place it in data/.
%
%   See also JMM.ACKNOWLEDGEMENT, JMM.DATADIR.
    if nargin < 1; tunes = {}; end
    if nargin < 2 || isempty(jsonPath)
        jsonPath = fullfile(jmm.dataDir(), 'ParseTrees.json');
    end
    if ~isfile(jsonPath)
        error('jmm:derivationsMissing', ...
            ['%s not found. The derivations are not distributed with the ' ...
             'toolbox; download ParseTrees.json from %s and place it at ' ...
             'that path (or pass its path).'], jsonPath, ...
            ['https://github.com/ren-zeng/formal-modeling-of-structural-' ...
             'repetition/blob/main/experiment/DataSet/Harmony/ParseTrees.json']);
    end
    corpus = jsondecode(fileread(jsonPath));

    % The file is a list of [name, tree] pairs, which jsondecode returns
    % as a cell array of two-element cells.
    nAll = numel(corpus);
    allNames = cell(1, nAll);
    for i = 1:nAll
        entry = corpus{i};
        allNames{i} = entry{1};
    end
    if isempty(tunes)
        wanted = 1:nAll;
    else
        tunes = cellstr(string(tunes));
        wanted = zeros(1, numel(tunes));
        for i = 1:numel(tunes)
            hit = find(strcmp(allNames, tunes{i}), 1);
            if isempty(hit)
                error('jmm:derivationsUnknownTune', ...
                    ['''%s'' is not in the corpus; it holds %d tunes, ' ...
                     'named like ''%s''.'], tunes{i}, nAll, allNames{1});
            end
            wanted(i) = hit;
        end
    end

    tuneCol = {};
    chordCol = [];
    levelCol = [];
    labelCol = {};
    for i = wanted
        entry = corpus{i};
        paths = localPaths(entry{2});
        for c = 1:numel(paths)
            labels = paths{c};
            for ell = 1:numel(labels)
                tuneCol{end + 1, 1} = allNames{i};       %#ok<AGROW>
                chordCol(end + 1, 1) = c - 1;            %#ok<AGROW>
                levelCol(end + 1, 1) = ell;              %#ok<AGROW>
                labelCol{end + 1, 1} = labels{ell};      %#ok<AGROW>
            end
        end
    end
    t = table(tuneCol, chordCol, levelCol, labelCol, ...
        'VariableNames', {'tune', 'chord', 'level', 'label'});
end


function out = localPaths(tree)
%LOCALPATHS  The root-to-leaf rule paths of one derivation, one per chord.
%   A node's contents are {chord, rule, children}; a leaf carries its chord
%   and no rule. The terminating rule directly above a leaf ends every path
%   and says nothing about structure, so it is dropped.
    out = {};
    out = localWalk(tree, {}, out);
end


function out = localWalk(node, path, out)
    if isfield(node, 'tag') && strcmp(node.tag, 'Leaf')
        out{end + 1} = path(~strcmp(path, 'Term'));
        return
    end
    contents = node.contents;
    rule = contents{2};
    if isfield(rule, 'contents') && (ischar(rule.contents) || isstring(rule.contents))
        label = char(rule.contents);
    else
        label = rule.tag;
    end
    children = contents{3};
    for c = 1:numel(children)
        if iscell(children)
            child = children{c};
        else
            child = children(c);
        end
        out = localWalk(child, [path, {label}], out);
    end
end
