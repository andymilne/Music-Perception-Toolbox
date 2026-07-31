function varargout = mptDefaults(varargin)
%MPTDEFAULTS  Get / set / reset toolbox-wide default options.
%
%   Centralises the user-tunable toolbox defaults (currently:
%   truncationSigmas, kernelPrecision, showHints, postHocGuards).
%   relAttrRoute (default 'auto') is a calibration and testing lever, not
%   part of the public interface: it pins the route a relative attribute
%   takes inside the Mobius method, which auto-dispatch otherwise chooses
%   on a cost estimate. 'auto' leaves that estimate in charge; 'centres'
%   forces the materialised tuple-centres route and 'grid' the
%   translation-grid route. Forcing a route overrides the cost judgement
%   only, never admissibility: see MOBIUS.MARELATTRPREFERSCENTRES.
%
%   orbitCostIntercept (default 3.8536) is the machine-specific term in
%   internal.orbitCostModel, which chooses between the Mobius and
%   enumerated combines. The default was measured on one machine; run
%   tools/calibrateOrbitIntercept to measure it on yours. Larger values
%   favour enumeration.
%
%   postHocGuards (default true) enables the checks that inspect a
%   route's output after computing it and may then recompute by another
%   route. Switch it off for calibration runs: with it on, the measured
%   cost of a route is not the cost of choosing it, because a diverting
%   check pays for both routes. showHints gates
%   one-time informational tips. Per-call name-value arguments always
%   override the defaults set here.
%
%   truncationSigmas (factory default 6) is the Gaussian kernel
%   truncation radius, in standard deviations. A kernel centred more
%   than truncationSigmas*sigma from an evaluation point is dropped; at
%   that radius the kernel has decayed to exp(-truncationSigmas^2/2) of
%   its peak. That factor is therefore an upper bound on the relative
%   contribution any single excluded centre would have made -- so the
%   absolute error of a kernel sum is at most exp(-truncationSigmas^2/2)
%   times the summed weight of the excluded centres, and its relative
%   error is of the same order wherever the retained centres dominate
%   (the usual case). Representative bounds (k = truncationSigmas):
%
%       truncationSigmas    error bound exp(-k^2/2)
%             3                    1.1e-2
%             4                    3.4e-4
%             5                    3.7e-6
%             6                    1.5e-8     (factory default)
%             7                    2.3e-11
%           Inf                    1.0e-12    (see below)
%
%   Inf does not sum into the far denormal tail. It resolves to the
%   finite width (~7.43 sigma) at which the kernel reaches the 1e-12
%   parity floor, applied uniformly across the absolute, relative,
%   single- and multi-attribute, kernel and orbit paths. Treat Inf as
%   "exact to 1e-12" rather than literally exhaustive.
%
%   What it affects. Every density evaluation routes through the
%   Gaussian kernel sum, so truncationSigmas governs the accuracy of
%   evalExpTens, cosSimExpTens, entropyExpTens, tensorHarmonicity,
%   templateHarmonicity, spectralEntropy, virtualPitches,
%   windowedTensorSimilarity, and weightEvents, together with the orbit
%   evaluators those functions call. It does not affect any non-kernel
%   computation.
%
%   Differential entropy is a special case. There truncationSigmas also
%   sets the adaptive convergence tolerance,
%   max(exp(-truncationSigmas^2/2), 1e-12), and hence how fine the
%   nested grid must become. At the tightest accuracy (Inf) a two- or
%   higher-dimensional grid can exceed the feasible size, in which case
%   entropyExpTens refuses with guidance rather than exhausting memory;
%   the factory default 6 keeps such grids feasible.
%
%   Call forms:
%
%     mptDefaults                       Print current values + a brief
%                                       summary of what each field means.
%     S = mptDefaults                   Return the current values as a
%                                       struct (no printing).
%     val = mptDefaults('name')         Return one value.
%     mptDefaults('name', val, ...)     Set one or more values.
%     prev = mptDefaults('name', val)   Set and capture the previous
%                                       values, for save-and-restore.
%     mptDefaults(prevStruct)           Restore from a previously-
%                                       returned struct. Inverse of the
%                                       setter form.
%     mptDefaults('reset')              Reset all to factory defaults.
%
%   Save-and-restore idiom (temporarily change defaults, then restore):
%
%     prev = mptDefaults('truncationSigmas', Inf, ...
%                        'kernelPrecision', 'single');
%     %  ... do work with the new defaults ...
%     mptDefaults(prev);     % restore exactly what was active before
%
%   Defaults persist within the MATLAB session but not across sessions.
%   `clear all` resets them.
%
%   Examples:
%
%     mptDefaults                                   % see current values
%     mptDefaults('truncationSigmas', Inf)          % exact (untruncated)
%     mptDefaults('truncationSigmas', Inf, ...
%                 'kernelPrecision', 'single')      % both at once
%     prev = mptDefaults('kernelPrecision', 'single');
%     ...
%     mptDefaults(prev)                             % restore
%     mptDefaults('reset')                          % back to factory
%
%   See also: INTERNAL.GAUSSIANKERNELSUM.

    persistent S
    if isempty(S)
        S = factoryDefaults();
    end

    if nargin == 0
        if nargout > 0
            varargout{1} = S;
        else
            printSummary(S);
        end
        return;
    end

    % Single-arg forms: 'reset' or a single field name.
    if nargin == 1
        arg = varargin{1};
        if isstruct(arg)
            % Restore from a previously-saved struct.
            old = S;
            fns = fieldnames(arg);
            for k = 1:numel(fns)
                S = setOne(S, fns{k}, arg.(fns{k}));
            end
            if nargout > 0
                varargout{1} = old;
            end
            return;
        end
        if ~(ischar(arg) || isstring(arg))
            error('mptDefaults:badArg', ...
                'Single argument must be a name, ''reset'', or a struct.');
        end
        name = char(arg);
        if strcmpi(name, 'reset')
            old = S;
            S = factoryDefaults();
            % Clear the dispatch-message throttle (which is normally
            % per-top-level-call via internal.dispatchScope) so the
            % next call sees its routing decision again, regardless
            % of where in a call tree the reset is issued from.
            internal.maybeShowDispatchMsg('reset');
            % Flush the kernelChunkBytes 'auto' resolution cache so
            % a subsequent call re-queries the OS.
            internal.kernelChunkBytesResolved('flushCache');
            if nargout > 0
                varargout{1} = old;
            end
            return;
        end
        if ~isfield(S, name)
            error('mptDefaults:unknownDefault', ...
                'Unknown default ''%s''. Valid names: %s.', ...
                name, strjoin(fieldnames(S), ', '));
        end
        varargout{1} = S.(name);
        if strcmpi(name, 'truncationSigmas')
            internal.maybeShowTruncationNotice();
        end
        return;
    end

    % Multi-arg: name-value pairs.
    if mod(nargin, 2) ~= 0
        error('mptDefaults:badNV', ...
            'Set form requires name-value pairs.');
    end
    old = S;
    for k = 1:2:nargin
        name = varargin{k};
        if ~(ischar(name) || isstring(name))
            error('mptDefaults:badNV', ...
                'Argument %d must be a name (string).', k);
        end
        S = setOne(S, char(name), varargin{k + 1});
    end
    if nargout > 0
        varargout{1} = old;
    end
end


function S = factoryDefaults()
    S = struct( ...
        'truncationSigmas', 6, ...
        'kernelPrecision', 'double', ...
        'showHints', true, ...
        'kernelChunkBytes', 'auto', ...
        'postHocGuards', true, ...
        'orbitCostIntercept', 3.8536, ...
        'relAttrRoute', 'auto' ...
    );
end


function printSummary(S)
%PRINTSUMMARY  Pretty-print current defaults with brief descriptions.
%   Called when mptDefaults is invoked at the prompt with no
%   arguments and no requested output. Programmatic callers
%   (S = mptDefaults) bypass this — they get the struct back
%   silently.

    if islogical(S.showHints) && S.showHints
        hintsStr = 'true';
    elseif islogical(S.showHints)
        hintsStr = 'false';
    else
        hintsStr = num2str(S.showHints);
    end

    fprintf('\nCurrent MPT defaults:\n\n');
    fprintf('  truncationSigmas: %-12s  Gaussian kernel truncation radius, in sigmas.\n', ...
            num2str(S.truncationSigmas));
    fprintf('                                  Inf = exact; larger is more accurate, slower.\n');
    fprintf('                                  Relative error exp(-k^2/2): 4 -> 3.4e-4,\n');
    fprintf('                                  5 -> 3.7e-6, 6 (default) -> 1.5e-8.\n');
    fprintf('  kernelPrecision : %-12s  Kernel-matrix arithmetic precision.\n', ...
            sprintf('''%s''', S.kernelPrecision));
    fprintf('                                  ''double'' (default) or ''single''.\n');
    fprintf('  showHints       : %-12s  Informational console messages from\n', ...
            hintsStr);
    fprintf('                                  the toolbox: dispatch decisions.\n');
    fprintf('                                  true or false.\n');
    if ischar(S.kernelChunkBytes)
        chunkStr = sprintf('''%s''', S.kernelChunkBytes);
    else
        chunkStr = sprintf('%g', S.kernelChunkBytes);
    end
    fprintf('  kernelChunkBytes: %-12s  Per-chunk byte budget for kernel-matrix\n', ...
            chunkStr);
    fprintf('                                  workloads. ''auto'' = 0.5 * available\n');
    fprintf('                                  physical memory; or a positive int.\n');
    fprintf('\n');
    fprintf('Usage:\n');
    fprintf('  mptDefaults(''name'', value)      set\n');
    fprintf('  prev = mptDefaults(...)         save previous values\n');
    fprintf('  mptDefaults(prev)               restore\n');
    fprintf('  mptDefaults(''reset'')            factory defaults\n');
    fprintf('  help mptDefaults                full help\n');
    fprintf('\n');
end


function S = setOne(S, name, value)
    switch lower(name)
        case 'truncationsigmas'
            if ~(isnumeric(value) && isscalar(value) && value > 0)
                error('mptDefaults:badValue', ...
                    '''truncationSigmas'' must be a positive scalar (Inf to disable).');
            end
            S.truncationSigmas = double(value);
        case 'kernelprecision'
            if ~(ischar(value) || isstring(value))
                error('mptDefaults:badValue', ...
                    '''kernelPrecision'' must be ''double'' or ''single''.');
            end
            v = lower(char(value));
            if ~ismember(v, {'double', 'single'})
                error('mptDefaults:badValue', ...
                    '''kernelPrecision'' must be ''double'' or ''single'' (got %s).', v);
            end
            S.kernelPrecision = v;
        case 'showhints'
            if ~(islogical(value) && isscalar(value))
                error('mptDefaults:badValue', ...
                    '''showHints'' must be true or false.');
            end
            S.showHints = value;
        case 'orbitcostintercept'
            if ~(isnumeric(value) && isscalar(value) && isfinite(value))
                error('mptDefaults:badValue', ...
                    '''orbitCostIntercept'' must be a finite scalar.');
            end
            S.orbitCostIntercept = double(value);
        case 'relattrroute'
            if ~(ischar(value) || isstring(value))
                error('mptDefaults:badValue', ...
                    ['''relAttrRoute'' must be ''auto'', ''centres'', ' ...
                     'or ''grid''.']);
            end
            v = lower(char(value));
            if ~ismember(v, {'auto', 'centres', 'grid'})
                error('mptDefaults:badValue', ...
                    ['''relAttrRoute'' must be ''auto'', ''centres'', ' ...
                     'or ''grid''; got ''%s''.'], char(value));
            end
            S.relAttrRoute = v;
        case 'posthocguards'
            if ~(islogical(value) && isscalar(value))
                error('mptDefaults:badValue', ...
                    '''postHocGuards'' must be true or false.');
            end
            S.postHocGuards = value;
        case 'kernelchunkbytes'
            if ischar(value) || isstring(value)
                v = lower(char(value));
                if ~strcmp(v, 'auto')
                    error('mptDefaults:badValue', ...
                        '''kernelChunkBytes'' string value must be ''auto''.');
                end
                S.kernelChunkBytes = v;
            elseif isnumeric(value) && isscalar(value) && value > 0 && ...
                    isfinite(value) && value == floor(value)
                S.kernelChunkBytes = double(value);
            else
                error('mptDefaults:badValue', ...
                    ['''kernelChunkBytes'' must be ''auto'' or a positive ' ...
                     'integer scalar (bytes).']);
            end
        otherwise
            error('mptDefaults:unknownDefault', ...
                'Unknown default ''%s''. Valid names: %s.', ...
                name, strjoin(fieldnames(S), ', '));
    end
end
