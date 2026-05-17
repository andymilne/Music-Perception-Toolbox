function varargout = mptDefaults(varargin)
%MPTDEFAULTS  Get / set / reset toolbox-wide default options.
%
%   Centralises the user-tunable toolbox defaults (currently:
%   truncationSigmas, kernelPrecision, showHints). The first two
%   are consulted by INTERNAL.GAUSSIANKERNELSUM and (via that helper)
%   by every centres-path consumer in the toolbox. showHints gates
%   one-time informational tips. Per-call name-value arguments
%   always override the defaults set here.
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
%     prev = mptDefaults('truncationSigmas', 6, ...
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
%     mptDefaults('truncationSigmas', 6)            % enable truncation
%     mptDefaults('truncationSigmas', 6, ...
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
        'truncationSigmas', Inf, ...
        'kernelPrecision', 'double', ...
        'showHints', true, ...
        'kernelChunkBytes', 'auto' ...
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
    fprintf('  truncationSigmas: %-12s  Gaussian kernel truncation in sigmas.\n', ...
            num2str(S.truncationSigmas));
    fprintf('                                  Inf = exact (default); 6 keeps\n');
    fprintf('                                  ~8 sig figs and is faster.\n');
    fprintf('  kernelPrecision : %-12s  Kernel-matrix arithmetic precision.\n', ...
            sprintf('''%s''', S.kernelPrecision));
    fprintf('                                  ''double'' (default) or ''single''.\n');
    fprintf('  showHints       : %-12s  Informational console messages from\n', ...
            hintsStr);
    fprintf('                                  the toolbox: kernel-eval tip and\n');
    fprintf('                                  dispatch decisions. true or false.\n');
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
