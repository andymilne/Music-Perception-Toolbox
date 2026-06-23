function g = resolveShape(shape)
%RESOLVESHAPE  Window shape -> gamma in [0,1] (0 Gaussian, 1 rectangle).
    if ischar(shape) || isstring(shape)
        switch lower(char(shape))
            case {'rect', 'rectangular', 'box'}, g = 1.0;
            case {'gaussian', 'gauss', 'normal'}, g = 0.0;
            otherwise
                error('mptWindowing:badShape', ...
                    ['Unknown window shape ''%s''; pass a number in [0,1] ' ...
                     '(0 Gaussian, 1 rectangular) or ''gaussian''/''rect''.'], char(shape));
        end
    else
        g = double(shape);
        if ~(g >= 0 && g <= 1)
            error('mptWindowing:badShape', 'Window shape must be in [0,1]; got %g.', g);
        end
    end
end
