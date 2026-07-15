%% diag_sa_flip.m — dump the field layout produced by the flipped SA build
%
%  After flipping localBuildSA to delegate to localBuildMA (A = 1), a
%  vector build returns a MaetDensity. This diagnostic builds a few SA
%  densities via the vector signature and prints every field name, class,
%  and size, plus what internal.saView exposes over each. It does NOT go
%  through the SA consumers (evalExpTens/cosSimExpTens/etc.), which are
%  not yet routed through saView and would error on the new tag.
%
%  Purpose: establish the exact produced layout so the view mapping and
%  consumer routing can be written against known shapes, not inferred.
%
%  Run: diag_sa_flip

fprintf('\n=== SA-flip field-layout diagnostic ===\n');

cases = {
  {'abs r2 K6 eager',  [0;3;7;11;14;18], [], 6.0, 2, false, false, 0,   false}
  {'rel r2 K6 eager',  [0;3;7;11;14;18], [], 30.0, 2, true,  true,  1200, false}
  {'abs r1 K5 eager',  [0;2;4;5;9],      [], 6.0, 1, false, false, 0,   false}
  {'abs r2 K6 lazy',   [0;3;7;11;14;18], [], 6.0, 2, false, false, 0,   true}
};

for ci = 1:numel(cases)
    c = cases{ci};
    [label, p, w, sigma, r, isRel, isPer, period, lazy] = c{:};
    fprintf('\n--- %s ---\n', label);
    if isempty(w), w = ones(numel(p), 1); end
    dens = buildExpTens(p, w, sigma, r, isRel, isPer, period, ...
        'lazy', lazy, 'verbose', false);

    fprintf('  tag = %s\n', dens.tag);
    iDumpFields(dens, '  dens');

    % saView over it
    try
        v = internal.saView(dens);
        fprintf('  --- internal.saView(dens) ---\n');
        iDumpFields(v, '  view');
    catch err
        fprintf('  saView ERRORED: %s (%s)\n', err.message, err.identifier);
    end
end

fprintf('\n=== end diagnostic ===\n');

% ---- helper ----
function iDumpFields(s, prefix)
    fns = fieldnames(s);
    for i = 1:numel(fns)
        f = fns{i};
        val = s.(f);
        cls = class(val);
        if iscell(val)
            szc = sprintf('cell{%s}', iSz(val));
            inner = '';
            if ~isempty(val)
                inner = sprintf(' inner1: %s %s', class(val{1}), iSz(val{1}));
            end
            fprintf('%s.%-12s : %s%s\n', prefix, f, szc, inner);
        elseif ischar(val)
            fprintf('%s.%-12s : char ''%s''\n', prefix, f, val);
        elseif isnumeric(val) || islogical(val)
            if isscalar(val)
                fprintf('%s.%-12s : %s scalar = %g\n', prefix, f, cls, double(val));
            else
                fprintf('%s.%-12s : %s %s\n', prefix, f, cls, iSz(val));
            end
        else
            fprintf('%s.%-12s : %s %s\n', prefix, f, cls, iSz(val));
        end
    end
end

function s = iSz(x)
    d = size(x);
    s = sprintf('%dx', d(1:end-1));
    s = [s, sprintf('%d', d(end))];
end
