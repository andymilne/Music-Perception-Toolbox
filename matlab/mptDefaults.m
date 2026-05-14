function varargout = mptDefaults(varargin)
%MPTDEFAULTS  Get / set / reset toolbox-wide default options.
%
%   This function centralises the few user-tunable toolbox defaults
%   (currently: truncationSigmas, kernelPrecision). The defaults are
%   consulted by INTERNAL.GAUSSIANKERNELSUM and (via that helper) by
%   every centres-path consumer in the toolbox. Per-call name-value
%   arguments always override the defaults set here.
%
%   Usage:
%
%     S = mptDefaults
%         Returns a struct of all current defaults.
%
%     val = mptDefaults('name')
%         Returns the current value of a single default.
%
%     mptDefaults('name', value, ...)
%         Sets one or more defaults. Multiple name-value pairs may be
%         given in a single call. Returns the previous values (as a
%         struct) so callers can restore them.
%
%     mptDefaults('reset')
%         Resets all defaults to their factory values:
%             truncationSigmas = Inf      (exact, backward-compatible)
%             kernelPrecision  = 'double' (backward-compatible)
%
%   Defaults persist within the MATLAB session but not across sessions.
%   `clear all` resets them.
%
%   Examples:
%
%     mptDefaults                                   % see current values
%     mptDefaults('truncationSigmas', 6)            % enable truncation
%     mptDefaults('truncationSigmas', 6, ...
%                 'kernelPrecision', 'single')             % both at once
%     prev = mptDefaults('kernelPrecision', 'single');     % save & restore idiom
%     ...
%     mptDefaults(prev)                              % restore
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
            disp(S);
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
        'showHints', true ...
    );
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
        otherwise
            error('mptDefaults:unknownDefault', ...
                'Unknown default ''%s''. Valid names: %s.', ...
                name, strjoin(fieldnames(S), ', '));
    end
end
