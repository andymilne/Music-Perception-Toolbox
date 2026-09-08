function pm = preMaet(pAttr, wAttr, specs)
%PREMAET Build a validated pre-MAET.
%
%   PM = PREMAET(PATTR) and PM = PREMAET(PATTR, WATTR, SPECS) return the
%   pre-MAET (Milne 2026, Def. 2.6) as a single struct with the fields
%   pAttr, wAttr, and specs. A pre-MAET is an event sequence together with
%   the elements each event contributes to each attribute and the
%   parameters that turn those elements into a density; those three parts
%   always travel together and always describe the same pre-MAET, so the
%   struct lets one variable hold the whole of it.
%
%   PM = PREMAET(PM0) validates an existing pre-MAET and returns a fresh
%   one; PM = PREMAET(PM0, WATTR, SPECS) replaces the parts given, leaving
%   the rest of PM0 in place.
%
%   It is a plain struct, not an object: its parts remain ordinary cells
%   and numerics, and every function that takes a pre-MAET equally takes
%   the three parts written out.
%
%   Inputs
%       pAttr - 1 x A cell of per-attribute value matrices, each K_a x N,
%               or an existing pre-MAET.
%       wAttr - [] for unweighted, a scalar applied to every attribute, or
%               a 1 x A cell whose entries are scalars, 1 x N vectors, or
%               K_a x N matrices. Optional, default [].
%       specs - [] or a 1 x A cell of per-attribute spec structs.
%               Optional, default [].
%
%   Output
%       pm - struct with fields pAttr, wAttr, and specs.
%
%   See also UNPACKPREMAET, SHOWPREMAET, FLATSPECS, BUILDEXPTENS.

if nargin < 2
    wAttr = [];
end
if nargin < 3
    specs = [];
end

if internal.isPreMaet(pAttr)
    base = pAttr;
    pAttr = base.pAttr;
    if isempty(wAttr) && ~iscell(wAttr)
        wAttr = base.wAttr;
    end
    if isempty(specs) && ~iscell(specs)
        specs = base.specs;
    end
elseif isstruct(pAttr)
    error('preMaet:badPAttr', ...
          ['A struct first argument must be a pre-MAET, with the ' ...
           'fields pAttr, wAttr, and specs.']);
end

if ~iscell(pAttr)
    error('preMaet:badPAttr', ...
          ['pAttr must be a cell of per-attribute value matrices. Wrap a ' ...
           'single attribute as {values}.']);
end
A = numel(pAttr);
if A < 1
    error('preMaet:noAttrs', 'pAttr must hold at least one attribute.');
end

wAttr = localCheckWeights(wAttr, A);
specs = localCheckSpecs(specs, A);

pm = struct('pAttr', {pAttr}, 'wAttr', {wAttr}, 'specs', {specs});
end


function w = localCheckWeights(w, A)
if isempty(w) && ~iscell(w)
    w = [];
    return;
end
if isnumeric(w) && isscalar(w)
    return;
end
if iscell(w)
    if numel(w) ~= A
        error('preMaet:badWeightLength', ...
              'wAttr must have length A = %d; got %d.', A, numel(w));
    end
    return;
end
error('preMaet:badWeightType', ...
      ['wAttr must be [], a scalar, or a 1 x A cell of per-attribute ' ...
       'weights.']);
end


function s = localCheckSpecs(s, A)
if isempty(s) && ~iscell(s)
    s = [];
    return;
end
if isstruct(s) && isscalar(s)
    error('preMaet:badSpecs', ...
          ['specs must be a 1 x A cell of per-attribute specs. Wrap a ' ...
           'single spec as {spec}.']);
end
if ~iscell(s)
    error('preMaet:badSpecs', ...
          'specs must be [] or a 1 x A cell of per-attribute specs.');
end
if numel(s) ~= A
    error('preMaet:badSpecsLength', ...
          'specs must have length A = %d; got %d.', A, numel(s));
end
end
