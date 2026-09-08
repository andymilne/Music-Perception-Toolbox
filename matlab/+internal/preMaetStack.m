function [P, W, tags] = preMaetStack(nodes, depth, a, name)
%PREMAETSTACK  Value, weight, and tag matrices for one attribute's cells.
%
%   Events may carry different numbers of elements, and a nested
%   attribute's groups may differ in size between events; both are
%   NaN-padded to the widest, the toolbox's convention for variable
%   cardinality. The tag matrix is reconstructed from the bracket
%   structure, so a file need never write it down.
%
%   See also READPREMAET, INTERNAL.PREMAETPARSECELL.
N = numel(nodes);
flat = cell(1, N);
for n = 1:N
    flat{n} = localFlatten(nodes{n}, depth);
end

if depth == 0
    K = 0;
    for n = 1:N
        K = max(K, size(flat{n}, 1));
    end
    paths = zeros(K, 0);
    slotPath = zeros(K, 0);
    slotPos = (1:K)';
else
    % One slot per (group path, position), sized by the widest event.
    allPaths = zeros(0, depth);
    for n = 1:N
        f = flat{n};
        for i = 1:size(f, 1)
            allPaths(end+1, :) = f(i, 1:depth); %#ok<AGROW>
        end
    end
    uniquePaths = unique(allPaths, 'rows');
    widest = zeros(size(uniquePaths, 1), 1);
    for n = 1:N
        f = flat{n};
        for u = 1:size(uniquePaths, 1)
            c = sum(all(f(:, 1:depth) == uniquePaths(u, :), 2));
            widest(u) = max(widest(u), c);
        end
    end
    slotPath = zeros(0, depth); slotPos = [];
    for u = 1:size(uniquePaths, 1)
        for i = 1:widest(u)
            slotPath(end+1, :) = uniquePaths(u, :); %#ok<AGROW>
            slotPos(end+1, 1) = i; %#ok<AGROW>
        end
    end
    K = size(slotPath, 1);
end

P = NaN(K, N); W = NaN(K, N); sawW = false;
for n = 1:N
    f = flat{n};
    used = zeros(K, 1);
    for i = 1:size(f, 1)
        if depth == 0
            k = i;
        else
            cand = find(all(slotPath == f(i, 1:depth), 2) & ~used);
            if isempty(cand)
                error('readPreMaet:cardinality', ...
                    ['Attribute %s: event %d has more elements in one ' ...
                     'group than any other event.'], localWho(a, name), n);
            end
            k = cand(1);
        end
        used(k) = 1;
        P(k, n) = f(i, depth + 1);
        if ~isnan(f(i, depth + 2))
            W(k, n) = f(i, depth + 2);
            sawW = true;
        end
    end
end
if ~sawW
    W = [];
end
if depth == 0
    tags = [];
else
    tags = slotPath;
end
end


function out = localFlatten(node, depth)
%LOCALFLATTEN  Leaves in reading order as [path..., value, weight] rows.
if depth == 0
    out = zeros(numel(node), 2);
    for i = 1:numel(node)
        out(i, :) = node{i};
    end
    return;
end
out = zeros(0, depth + 2);
for g = 1:numel(node)
    sub = localFlatten(node{g}, depth - 1);
    for i = 1:size(sub, 1)
        out(end+1, :) = [g - 1, sub(i, :)]; %#ok<AGROW>
    end
end
end


function who = localWho(a, name)
if isempty(name)
    who = sprintf('%d', a);
else
    who = name;
end
end
