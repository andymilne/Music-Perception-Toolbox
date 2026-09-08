function [pAttr, wAttr, specs, rest] = preMaetArgs(args)
%PREMAETARGS  Split a pre-MAET call's arguments into its leading parts.
%
%   The pre-MAET functions accept either a whole pre-MAET followed by the
%   remaining arguments, or pAttr and wAttr followed by those same
%   remaining arguments. This front end returns the three parts of the
%   pre-MAET and the arguments that follow it. Where the call passed the
%   parts, specs is [] and the callee's own 'specs' name-value stands.
%
%   Input
%       args - 1 x n cell of the arguments as received.
%
%   Outputs
%       pAttr, wAttr, specs - the three parts of the pre-MAET.
%       rest                - 1 x m cell of the remaining arguments.

if isempty(args)
    error('mpt:preMaetArgs:noInput', ...
          'A pre-MAET is required as the first argument.');
end

if internal.isPreMaet(args{1})
    pm = args{1};
    pAttr = pm.pAttr;
    wAttr = pm.wAttr;
    specs = pm.specs;
    rest = args(2:end);
    return;
end

if numel(args) < 2
    error('mpt:preMaetArgs:noWeights', ...
          ['Called with the loose triple, pAttr must be followed by ' ...
           'wAttr. Pass [] for unweighted, or pass a whole pre-MAET.']);
end
pAttr = args{1};
wAttr = args{2};
specs = [];
rest = args(3:end);
end
