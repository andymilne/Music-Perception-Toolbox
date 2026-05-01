function wmd = windowTensor(dens, windowSpec)
%WINDOWTENSOR  Wrap a MaetDensity with a post-tensor window specification.
%
%   wmd = windowTensor(dens, windowSpec) returns a WindowedMaetDensity
%   that bundles the underlying MAET density with a window spec. No
%   math is performed at construction time; the window is applied
%   lazily by evalExpTens (pointwise multiplication by the window
%   function) and by cosSimExpTens (closed-form windowed inner
%   product). See the MAET specification §4.3 for the full semantics.
%
%   Inputs
%       dens        - MaetDensity (from buildExpTens, MA call form).
%       windowSpec  - Struct with fields:
%           size    - Per-group window effective standard deviation in
%                     multiples of that group's sigma. Scalar (broadcast
%                     across all groups) or 1 x G vector. NaN or Inf on
%                     an entry means the group is not windowed.
%           mix     - Per-group shape parameter in [0, 1]: 0 = pure
%                     Gaussian, 1 = pure rectangular, in between =
%                     rectangular-convolved-with-Gaussian. Scalar or
%                     1 x G vector.
%           centre  - 1 x A cell of per-attribute centre coordinates,
%                     each a column vector of length
%                     dim_per_attr(a); concatenating gives the centre
%                     point in the group's effective subspace.
%                     Alternatively, a flat dim x 1 vector will be
%                     split by dim_per_attr. Optional: defaults to
%                     zero for all attributes.
%
%   Output
%       wmd - WindowedMaetDensity struct (tagged 'WindowedMaetDensity').
%
%   See also buildExpTens, evalExpTens, cosSimExpTens, windowedSimilarity.

    if ~isstruct(dens) || ~isfield(dens, 'tag') || ...
            ~strcmp(dens.tag, 'MaetDensity')
        error('windowTensor:badDens', ...
              'dens must be a MaetDensity struct.');
    end
    if ~isstruct(windowSpec)
        error('windowTensor:badSpec', 'windowSpec must be a struct.');
    end

    A = dens.nAttrs;
    G = dens.nGroups;
    dimPerAttr = dens.dimPerAttr;
    dim_total = dens.dim;

    % --- size ---
    if ~isfield(windowSpec, 'size')
        error('windowTensor:missingSize', ...
              'windowSpec must contain a ''size'' field.');
    end
    size_arr = double(windowSpec.size(:).');
    if isscalar(size_arr)
        size_arr = repmat(size_arr, 1, G);
    end
    if numel(size_arr) ~= G
        error('windowTensor:sizeLength', ...
              'windowSpec.size must be a scalar or length-%d vector; got length %d.', ...
              G, numel(size_arr));
    end

    % --- mix ---
    if ~isfield(windowSpec, 'mix')
        error('windowTensor:missingMix', ...
              'windowSpec must contain a ''mix'' field.');
    end
    mix_arr = double(windowSpec.mix(:).');
    if isscalar(mix_arr)
        mix_arr = repmat(mix_arr, 1, G);
    end
    if numel(mix_arr) ~= G
        error('windowTensor:mixLength', ...
              'windowSpec.mix must be a scalar or length-%d vector; got length %d.', ...
              G, numel(mix_arr));
    end
    if any(mix_arr < 0) || any(mix_arr > 1)
        error('windowTensor:mixRange', ...
              'windowSpec.mix entries must be in [0, 1].');
    end

    % --- centre ---
    centre_list = cell(1, A);
    if ~isfield(windowSpec, 'centre') || isempty(windowSpec.centre)
        for a = 1:A
            centre_list{a} = zeros(dimPerAttr(a), 1);
        end
    elseif iscell(windowSpec.centre)
        if numel(windowSpec.centre) ~= A
            error('windowTensor:centreLength', ...
                  'windowSpec.centre (cell form) must have length %d; got %d.', ...
                  A, numel(windowSpec.centre));
        end
        for a = 1:A
            c = double(windowSpec.centre{a}(:));
            if numel(c) ~= dimPerAttr(a)
                error('windowTensor:centreAttrLength', ...
                      'windowSpec.centre{%d} must have length %d; got %d.', ...
                      a, dimPerAttr(a), numel(c));
            end
            centre_list{a} = c;
        end
    else
        flat = double(windowSpec.centre(:));
        if numel(flat) ~= dim_total
            error('windowTensor:centreFlatLength', ...
                  'windowSpec.centre (flat vector form) must have length dim = %d; got %d.', ...
                  dim_total, numel(flat));
        end
        offset = 0;
        for a = 1:A
            da = dimPerAttr(a);
            centre_list{a} = flat(offset + 1 : offset + da);
            offset = offset + da;
        end
    end

    wmd = struct();
    wmd.tag    = 'WindowedMaetDensity';
    wmd.dens   = dens;
    wmd.size   = size_arr;
    wmd.mix    = mix_arr;
    wmd.centre = centre_list;
end
