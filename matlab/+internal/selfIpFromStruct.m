function cache = selfIpFromStruct(d)
%INTERNAL.SELFIPFROMSTRUCT  Self-IP memo cache carried by a density struct.
%
%   CACHE = INTERNAL.SELFIPFROMSTRUCT(D) reads the memo cache from D's
%   'selfIP' field, or returns an empty cache when the field is absent
%   or malformed (an unrecognised shape simply recomputes; it can never
%   produce a wrong value). Shared by cosSimExpTens, which seeds its
%   per-call caches from the operand structs, and by
%   INTERNAL.FLATSELECTORINPUTS, which reads the same caches when
%   explainDispatch asks what the call would price.
%
%   See also INTERNAL.SELFIPMEMOISED, INTERNAL.SELFIPKEY.

    if isstruct(d) && isfield(d, 'selfIP') && isstruct(d.selfIP) ...
            && isfield(d.selfIP, 'keys') && isfield(d.selfIP, 'vals') ...
            && iscell(d.selfIP.keys) ...
            && numel(d.selfIP.keys) == numel(d.selfIP.vals)
        cache = d.selfIP;
    else
        cache = struct('keys', {{}}, 'vals', []);
    end
end
