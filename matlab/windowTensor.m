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
%                     across all attributes) or 1 x A vector. NaN or Inf on
%                     an entry means the group is not windowed.
%           mix     - Per-group shape parameter in [0, 1]: 0 = pure
%                     Gaussian, 1 = pure rectangular, in between =
%                     rectangular-convolved-with-Gaussian. Scalar or
%                     1 x A vector.
%           centre  - Per-attribute centre coordinates, in any of three
%                     forms:
%                       * Numeric scalar (or 1x1 numeric array, or any
%                         numeric size-1 ndarray): broadcast to fill
%                         every per-attribute slot uniformly.
%                         Convenient when one window centre is wanted
%                         everywhere.
%                       * 1 x A cell, each entry a column vector of
%                         length dim_per_attr(a) (cell form).
%                       * Flat dim x 1 numeric vector, split by
%                         dim_per_attr in attribute order (flat form).
%                     Cell inputs are always interpreted structurally
%                     and never broadcast: a length-1 cell on an A > 1
%                     density raises rather than silently broadcasting.
%                     Optional: defaults to zero for all attributes.
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
    dimPerAttr = dens.dimPerAttr;
    dim_total = dens.dim;

    % --- size ---
    if ~isfield(windowSpec, 'size')
        error('windowTensor:missingSize', ...
              'windowSpec must contain a ''size'' field.');
    end
    size_arr = double(windowSpec.size(:).');
    if isscalar(size_arr)
        size_arr = repmat(size_arr, 1, A);
    end
    if numel(size_arr) ~= A
        error('windowTensor:sizeLength', ...
              'windowSpec.size must be a scalar or length-%d vector; got length %d.', ...
              A, numel(size_arr));
    end

    % --- mix ---
    if ~isfield(windowSpec, 'mix')
        error('windowTensor:missingMix', ...
              'windowSpec must contain a ''mix'' field.');
    end
    mix_arr = double(windowSpec.mix(:).');
    if isscalar(mix_arr)
        mix_arr = repmat(mix_arr, 1, A);
    end
    if numel(mix_arr) ~= A
        error('windowTensor:mixLength', ...
              'windowSpec.mix must be a scalar or length-%d vector; got length %d.', ...
              A, numel(mix_arr));
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
        % Cell inputs are always interpreted structurally as cell-form
        % (length must equal A). A length-1 cell on an A > 1 density
        % raises rather than silently broadcasting; this preserves the
        % pre-fix contract that a wrong-length cell is a user error.
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
    elseif isnumeric(windowSpec.centre)
        % Numeric input. Two interpretations:
        %   * numel == 1 — scalar broadcast: fill every per-attribute
        %     slot uniformly with the scalar value. Covers ordinary
        %     scalars (``5.0``), 1x1 arrays, and any numeric size-1
        %     ndarray. Convenient when one window centre is wanted
        %     everywhere.
        %   * numel == dim_total — flat form, split by dim_per_attr.
        %   * any other length — error.
        % Cell inputs do NOT take this path; they are interpreted
        % structurally above. A wrong-length cell raises rather than
        % silently broadcasting.
        flat = double(windowSpec.centre(:));
        if numel(flat) == 1
            val = flat;
            for a = 1:A
                centre_list{a} = repmat(val, dimPerAttr(a), 1);
            end
        elseif numel(flat) == dim_total
            offset = 0;
            for a = 1:A
                da = dimPerAttr(a);
                centre_list{a} = flat(offset + 1 : offset + da);
                offset = offset + da;
            end
        else
            error('windowTensor:centreFlatLength', ...
                  'windowSpec.centre (flat vector form) must have length 1 (scalar broadcast) or dim = %d; got %d.', ...
                  dim_total, numel(flat));
        end
    else
        error('windowTensor:centreType', ...
              'windowSpec.centre must be numeric or a cell of per-attribute vectors; got %s.', ...
              class(windowSpec.centre));
    end

    wmd = struct();
    wmd.tag    = 'WindowedMaetDensity';
    wmd.dens   = dens;
    wmd.size   = size_arr;
    wmd.mix    = mix_arr;
    wmd.centre = centre_list;
end
