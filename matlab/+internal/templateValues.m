function [tmpl_vals, tmpl_norm_sq, margin] = templateValues(spectrum, ...
        sigma, step, per, period, truncationSigmas, kernelPrecision)
%TEMPLATEVALUES  Evaluate the harmonic template on its grid.
%
%   [tmpl_vals, tmpl_norm_sq, margin] = internal.templateValues( ...
%       spectrum, sigma, step, per, period, truncationSigmas, kernelPrecision)
%
%   spectrum is the cell of addSpectra arguments defining the template
%   (one tone at pitch 0, weight 1). Non-periodic: the grid runs from
%   -margin to the top partial plus margin, margin being the
%   truncation radius k * sigma at the resolved truncationSigmas k.
%   Periodic: the template is folded into one period (each partial's
%   Gaussian summed over its periodic images) and evaluated on
%   internal.periodicGrid(period, step); margin is 0.
%
%   Shared by templateHarmonicity and virtualPitches (scalar and
%   batched paths).
%
%   See also INTERNAL.TEMPLATEXCORRCHORDSIDE, INTERNAL.PERIODICGRID.

    [tmpl_p, tmpl_w] = addSpectra(0, 1, spectrum{:});
    if per
        tmpl_dens = buildMaet(tmpl_p, tmpl_w, sigma, 1, false, ...
            true, period, 'verbose', false);
        x_tmpl = internal.periodicGrid(period, step);
        margin = 0;
    else
        tmpl_dens = buildMaet(tmpl_p, tmpl_w, sigma, 1, false, ...
            false, 1200, 'verbose', false);
        margin = internal.accuracyFloor('resolve', truncationSigmas) * sigma;
        x_tmpl = -margin:step:(max(tmpl_p) + margin);
    end
    tmpl_vals = evalMaet(tmpl_dens, x_tmpl, ...
        'truncationSigmas', truncationSigmas, ...
        'kernelPrecision', kernelPrecision, ...
        'verbose', false);
    tmpl_norm_sq = sum(tmpl_vals .^ 2);
end
