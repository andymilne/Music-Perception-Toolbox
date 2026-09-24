function pm = boundContext(L, rInner, flag)
%BOUNDCONTEXT  Every window of L consecutive beats, bound and nested.
%
%   pm = jmm.boundContext(L)
%   pm = jmm.boundContext(L, rInner)
%   pm = jmm.boundContext(L, rInner, flagName)
%
%   bindEvents slides the window along the whole chorale at once, so the
%   piece is nested once rather than once per position: the result holds one
%   super-event per window, the inner level each beat's pitch multiset
%   ([exch] = 1, r = rInner, default 1) and the outer level the L ordered
%   beats ([exch] = 0, r = L), relative at the outer level alone
%   ([rel] = (0, 1)) and periodic at the octave.
%
%   Per-attribute bind orders keep everything but the pitch flat at one
%   value per window: the window's own start time, which locates it for a
%   sweep, and the named inversion flag ('sixFour' or 'rootPositionNext')
%   where one is asked for.
%
%   See also BINDEVENTS, JMM.QUERY, JMM.BWVWINDOWSTATE.
    if nargin < 2 || isempty(rInner), rInner = 1; end
    if nargin < 3, flag = ''; end
    S = jmm.bwvWindowState();
    [pAttr, wAttr, specs] = unpackPreMaet(S.beatsPm);
    specs{1}.r = rInner; specs{1}.rel = false; specs{1}.exch = true;
    orders = [L, 1];
    if ~isempty(flag)
        values = S.flags.(flag);
        pAttr{end + 1} = values;
        wAttr{end + 1} = ones(size(values));
        fs = flatSpecs({values}, 'r', 1, 'rel', false, 'exch', false, ...
                       'name', 'flag', 'sigma', S.sigmaFlag, ...
                       'isPer', false, 'period', 0.0);
        specs{end + 1} = fs{1};
        orders(end + 1) = 1;
    end
    pm = bindEvents(pAttr, wAttr, orders, 'relOuter', true, 'specs', specs);
end
