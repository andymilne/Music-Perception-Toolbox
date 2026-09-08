function args = windowedPreMaetArgs(args, fname, nPre)
%WINDOWEDPREMAETARGS  Expand pre-MAET arguments into the positional form.
%
%   windowedSimilarity and windowedEntropy take their geometry
%   positionally, as vectors shared by both operands. Given whole
%   pre-MAETs instead, this reads the shared geometry out of their specs,
%   applies any of the six per-attribute overrides passed as name-value
%   arguments, and returns the call in the positional form the workers
%   already speak.
%
%   The two pre-MAETs of windowedSimilarity describe one comparison, so
%   they must agree on the structural geometry: same attribute count, and
%   the same r, [rel], [sym], and nesting on every attribute. The context
%   supplies the specs; a disagreement is an error rather than a silent
%   choice between them.
%
%   Inputs
%       args  - 1 x n cell of the arguments as received.
%       fname - caller's name, for error messages.
%       nPre  - number of pre-MAET operands the caller takes (2 for
%               windowedSimilarity, 1 for windowedEntropy).
%
%   Output
%       args - the arguments with each pre-MAET replaced by its pAttr and
%              wAttr, the six geometry vectors inserted after them, and
%              the overriding name-value arguments removed.

if numel(args) < nPre || ~internal.isPreMaet(args{1})
    return;                                  % positional form; unchanged
end
for k = 2:nPre
    if ~internal.isPreMaet(args{k})
        error([fname ':mixedPreMaet'], ...
              ['Given a pre-MAET as the first argument, argument %d ' ...
               'must be a pre-MAET too: the two describe one ' ...
               'comparison and are passed the same way.'], k);
    end
end

pms = args(1:nPre);
rest = args(nPre+1:end);
specs = pms{1}.specs;
if isempty(specs)
    error([fname ':noSpecs'], ...
          ['A pre-MAET passed here must carry its specs: they are ' ...
           'where the shared geometry is read from. Build it with ' ...
           'preMaet(pAttr, wAttr, specs), or use the positional form.']);
end
A = numel(pms{1}.pAttr);
for k = 2:nPre
    localCheckAgrees(specs, pms{k}.specs, A, fname);
end

% Pull the six overrides out of the trailing name-value arguments; what
% is left is passed on untouched.
[kw, rest] = localTakeOverrides(rest);

specs = internal.overrideSpecs(specs, kw.r, kw.rel, kw.sym, A);
[~, ~, ~, ~, names, specKernel] = internal.normaliseSpecs(specs, A);
sigma  = internal.resolveKernelParam(kw.sigma,  specKernel.sigma,  ...
                                     'sigma',  names, A, false);
isPer  = internal.resolveKernelParam(kw.isPer,  specKernel.isPer,  ...
                                     'isPer',  names, A, false);
period = internal.resolveKernelParam(kw.period, specKernel.period, ...
                                     'period', names, A, true);
[rVec, isRelVec, isSymVec, nestedList] = internal.normaliseSpecs(specs, A);

parts = cell(1, 2 * nPre);
for k = 1:nPre
    parts{2*k - 1} = pms{k}.pAttr;
    parts{2*k}     = pms{k}.wAttr;
end
args = [parts, {sigma, rVec, isRelVec, isPer, period}, rest];
if any(~cellfun(@isempty, nestedList))
    args = [args, {'specs', specs}];         % nested geometry travels on
else
    args = [args, {'isSym', isSymVec}];
end
end


function [kw, rest] = localTakeOverrides(rest)
%LOCALTAKEOVERRIDES  Remove the six per-attribute overrides from rest.
    kw = struct('sigma', [], 'isPer', [], 'period', [], ...
                'r', [], 'rel', [], 'sym', []);
    keys = fieldnames(kw);
    keep = true(1, numel(rest));
    i = 1;
    while i <= numel(rest)
        nm = rest{i};
        if (ischar(nm) || (isstring(nm) && isscalar(nm))) && i < numel(rest)
            hit = find(strcmpi(char(nm), keys), 1);
            if ~isempty(hit)
                kw.(keys{hit}) = rest{i + 1};
                keep(i) = false; keep(i + 1) = false;
                i = i + 2;
                continue;
            end
        end
        i = i + 1;
    end
    rest = rest(keep);
end


function localCheckAgrees(specsA, specsB, A, fname)
%LOCALCHECKAGREES  The two pre-MAETs must share the structural geometry.
    if isempty(specsB) || numel(specsB) ~= A
        error([fname ':specsMismatch'], ...
              ['The two pre-MAETs must have the same attribute count ' ...
               'and both carry specs: they describe one comparison.']);
    end
    for a = 1:A
        for f = {'r', 'rel', 'sym', 'tags'}
            va = localField(specsA{a}, f{1});
            vb = localField(specsB{a}, f{1});
            if ~isequaln(double(va(:)).', double(vb(:)).')
                error([fname ':specsMismatch'], ...
                      ['The two pre-MAETs disagree on ''%s'' for ' ...
                       'attribute %d. They describe one comparison, so ' ...
                       'the structural geometry must match; sigma, ' ...
                       'isPer and period may differ and are taken from ' ...
                       'the first.'], f{1}, a);
            end
        end
    end
end


function v = localField(s, f)
    if isstruct(s) && isfield(s, f)
        v = s.(f);
    else
        v = [];
    end
end
