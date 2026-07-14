function varargout = accuracyFloor(cmd, arg)
%ACCURACYFLOOR  Toolbox accuracy-floor state and truncation resolver.
%
%   Mirrors the Python mpt._defaults accuracy-floor machinery so the two
%   languages resolve the Inf ("exact") truncation sentinel identically.
%   Inf does not mean literally exhaustive summation into the denormal
%   far tail; it means "accuracy-floor accuracy": a kernel is summed out
%   to the finite width at which it falls below EPS (default 1e-12), the
%   toolbox parity floor. This makes Inf uniform across the absolute and
%   relative, single- and multi-attribute, kernel and orbit paths.
%
%   Commands:
%
%     eps    = internal.accuracyFloor('eps')
%              Current floor epsilon (the override value if one is
%              active, else the 1e-12 default).
%
%     k      = internal.accuracyFloor('sigmas')
%              Finite truncation width k = sqrt(-2 ln eps) at which a
%              unit Gaussian kernel falls to the current eps. The single
%              point that turns the floor epsilon into a sigma count.
%
%     ts     = internal.accuracyFloor('resolve', truncationSigmas)
%              Resolve a truncation knob to an effective finite width:
%              [] -> the mptDefaults('truncationSigmas') default; a
%              non-finite value (Inf) -> the 'sigmas' width above; a
%              finite positive value passes through unchanged. Every
%              truncation site calls this so the resolution is uniform.
%
%     prev   = internal.accuracyFloor('setEps', eps)
%              Override the floor epsilon (in (0, 1)); returns the
%              previous value for save-and-restore. Chiefly for
%              golden-value regeneration at maximal accuracy, e.g.
%              eps = 1e-300 (~37 sigma, effectively exhaustive). Pair
%              with a restore in an onCleanup guard:
%
%                prev = internal.accuracyFloor('setEps', 1e-300);
%                cleanup = onCleanup(@() internal.accuracyFloor( ...
%                                          'setEps', prev));
%                %  ... regenerate goldens at the wider width ...
%
%              This is not a user-facing default (not in mptDefaults);
%              it overrides the floor EPS, not the width directly.
%
%     internal.accuracyFloor('resetEps')
%              Clear any override, restoring the 1e-12 default.

    persistent EPS_OVERRIDE
    DEFAULT_EPS = 1e-12;

    switch cmd
        case 'eps'
            if isempty(EPS_OVERRIDE)
                varargout{1} = DEFAULT_EPS;
            else
                varargout{1} = EPS_OVERRIDE;
            end

        case 'sigmas'
            if isempty(EPS_OVERRIDE)
                e = DEFAULT_EPS;
            else
                e = EPS_OVERRIDE;
            end
            varargout{1} = sqrt(-2.0 * log(e));

        case 'resolve'
            if nargin < 2 || isempty(arg)
                ts = mptDefaults('truncationSigmas');
            else
                ts = double(arg);
            end
            if ~isfinite(ts)
                if isempty(EPS_OVERRIDE)
                    e = DEFAULT_EPS;
                else
                    e = EPS_OVERRIDE;
                end
                varargout{1} = sqrt(-2.0 * log(e));
            else
                varargout{1} = ts;
            end

        case 'setEps'
            if isempty(EPS_OVERRIDE)
                prev = DEFAULT_EPS;
            else
                prev = EPS_OVERRIDE;
            end
            if nargin < 2 || isempty(arg)
                error('mpt:accuracyFloor:missingEps', ...
                    '''setEps'' requires an epsilon in (0, 1).');
            end
            e = double(arg);
            if ~(e > 0 && e < 1)
                error('mpt:accuracyFloor:badEps', ...
                    'Accuracy-floor eps must be in (0, 1); got %g.', e);
            end
            EPS_OVERRIDE = e;
            varargout{1} = prev;

        case 'resetEps'
            EPS_OVERRIDE = [];

        otherwise
            error('mpt:accuracyFloor:badCmd', ...
                'Unknown command ''%s''.', cmd);
    end
end
