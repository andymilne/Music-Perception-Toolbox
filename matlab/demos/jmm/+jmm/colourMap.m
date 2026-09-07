function map = colourMap(name, n)
%COLOURMAP  The matplotlib 'viridis' and 'magma' colour maps, for the figures.
%
%   map = jmm.colourMap(name)
%   map = jmm.colourMap(name, n)
%
%   Returns an n x 3 (default 256) colour map interpolated from seventeen
%   anchor colours of matplotlib's perceptually uniform maps, so the
%   MATLAB figures use the same maps as the Python demos without depending
%   on any colour map that is present in one of MATLAB and Octave only.
%   name is 'viridis' or 'magma'.
    if nargin < 2 || isempty(n), n = 256; end
    switch lower(name)
        case 'viridis'
            A = [0.2670 0.0049 0.3294;
                 0.2823 0.0950 0.4173;
                 0.2788 0.1755 0.4834;
                 0.2590 0.2515 0.5247;
                 0.2297 0.3224 0.5457;
                 0.1994 0.3876 0.5546;
                 0.1727 0.4488 0.5579;
                 0.1490 0.5081 0.5573;
                 0.1276 0.5669 0.5506;
                 0.1206 0.6258 0.5335;
                 0.1579 0.6838 0.5017;
                 0.2461 0.7389 0.4520;
                 0.3692 0.7889 0.3829;
                 0.5160 0.8312 0.2943;
                 0.6785 0.8637 0.1895;
                 0.8456 0.8873 0.0997;
                 0.9932 0.9062 0.1439];
        case 'magma'
            A = [0.0015 0.0005 0.0139;
                 0.0396 0.0311 0.1335;
                 0.1131 0.0655 0.2768;
                 0.2117 0.0620 0.4186;
                 0.3167 0.0717 0.4854;
                 0.4147 0.1104 0.5047;
                 0.5128 0.1482 0.5076;
                 0.6136 0.1818 0.4985;
                 0.7164 0.2150 0.4753;
                 0.8169 0.2559 0.4365;
                 0.9043 0.3196 0.3881;
                 0.9609 0.4183 0.3596;
                 0.9867 0.5356 0.3822;
                 0.9961 0.6537 0.4462;
                 0.9969 0.7696 0.5349;
                 0.9924 0.8843 0.6401;
                 0.9871 0.9914 0.7495];
        otherwise
            error('jmm:colormap', 'Unknown colour map ''%s''.', name);
    end
    x = linspace(0, 1, size(A, 1));
    map = interp1(x, A, linspace(0, 1, n), 'linear');
end
