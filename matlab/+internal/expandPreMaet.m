function args = expandPreMaet(args, idx)
%EXPANDPREMAET  Replace a pre-MAET at position idx with its three parts.
%   For the functions whose positional arguments are pAttr, w, and specs in
%   that order: a whole pre-MAET given in their place expands into exactly
%   those three, so the rest of the call is unchanged.
    if numel(args) >= idx && internal.isPreMaet(args{idx})
        pm = args{idx};
        args = [args(1:idx-1), {pm.pAttr, pm.wAttr, pm.specs}, ...
                args(idx+1:end)];
    end
end
