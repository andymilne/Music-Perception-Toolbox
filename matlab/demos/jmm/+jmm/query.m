function pm = query(chords, flag, rInner)
%QUERY  A literal chord succession as a bound query.
%
%   pm = jmm.query(chords)
%   pm = jmm.query(chords, flag)
%   pm = jmm.query(chords, flag, rInner)
%
%   chords is a 1 x L cell of MIDI pitch vectors. They are read exactly as
%   a window of the chorale is: the same conversion, the same bind orders,
%   and a constant flag value ([] for none) where the context carries one.
%
%   See also JMM.BOUNDCONTEXT, PREMAETFROMATTRTABLE.
    if nargin < 2, flag = []; end
    if nargin < 3 || isempty(rInner), rInner = 1; end
    S = jmm.bwvWindowState();
    onset = []; pitch = [];
    for j = 1:numel(chords)
        c = double(chords{j}(:));
        onset = [onset; (j - 1) * ones(numel(c), 1)]; %#ok<AGROW>
        pitch = [pitch; c]; %#ok<AGROW>
    end
    t = table(onset, pitch, ones(numel(pitch), 1), ...
              'VariableNames', {'onsetBeats', 'pitch', 'weight'});
    flat = preMaetFromAttrTable(t, 'attributes', { ...
        struct('column', 'pitch', 'sigma', S.sigmaPitch, 'r', 1, ...
               'exch', true, 'isPer', true, 'period', S.period), ...
        struct('column', 'onset', 'sigma', 1.0)}, ...
        'time', 'beats', 'weights', 'weight');
    [pAttr, wAttr, specs] = unpackPreMaet(flat);
    specs{1}.r = rInner; specs{1}.rel = false; specs{1}.exch = true;
    orders = [numel(chords), 1];
    if ~isempty(flag)
        values = double(flag) * ones(1, numel(chords));
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
