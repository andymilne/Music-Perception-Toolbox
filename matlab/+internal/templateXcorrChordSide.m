function xcorr_norm = templateXcorrChordSide( ...
        chord_p, chord_w, sigma, ...
        tmpl_vals, tmpl_norm_sq, margin, step, ...
        truncationSigmas, kernelPrecision, per, period)
%TEMPLATEXCORRCHORDSIDE  Normalised cross-correlation profile of a chord
%against a pre-evaluated template spectrum.
%
%   xcorr_norm = internal.templateXcorrChordSide( ...
%       chord_p, chord_w, sigma, ...
%       tmpl_vals, tmpl_norm_sq, margin, step, ...
%       truncationSigmas, kernelPrecision)
%   xcorr_norm = internal.templateXcorrChordSide(..., per, period)
%
%   Builds a 1-D absolute non-periodic expectation tensor from
%   (chord_p, chord_w) under Gaussian smoothing sigma, evaluates it
%   on the grid -margin:step:(max(chord_p) + margin), convolves with the
%   reverse of the supplied template values, and normalises by
%   sqrt(sum(chord_vals.^2) * tmpl_norm_sq) so each lag is a cosine
%   similarity in [0, 1].
%
%   With per = true, the chord's spectrum is folded into one period
%   (each partial's Gaussian summed over its periodic images),
%   evaluated on the periodic grid 0:step:(period - step) (see
%   internal.periodicGrid), and cross-correlated circularly with the
%   template, which must have been evaluated on the same grid
%   (internal.templateValues).
%
%   Shared by templateHarmonicity (which returns max + optional
%   Harrison-2020 entropy of the profile) and virtualPitches (which
%   returns the profile re-packaged as (vp_p, vp_w)). Hoisting this
%   into +internal lets the batched paths of both functions share a
%   pre-evaluated template and use this same per-row chord-side
%   compute, identically.
%
%   Inputs:
%     chord_p          - Pitch row vector (cents).
%     chord_w          - Weights (same length as chord_p).
%     sigma            - Gaussian smoothing width (cents).
%     tmpl_vals        - Pre-evaluated template values on its own grid.
%     tmpl_norm_sq     - sum(tmpl_vals.^2) (caller pre-computes once).
%     margin           - Grid margin in cents, the truncation radius
%                        k * sigma at the resolved truncationSigmas k,
%                        applied below the lowest pitch and above the
%                        highest. The template's grid must start at the
%                        same -margin, so that each lag index maps to
%                        the same pitch offset.
%     step             - Grid spacing in cents (typically 1).
%     truncationSigmas - Forwarded to evalMaet.
%     kernelPrecision  - Forwarded to evalMaet.
%     per              - Logical (default false). Periodic mode.
%     period           - Period in cents when per (default 1200).
%
%   Output:
%     xcorr_norm       - Column vector of normalised cross-correlation
%                        values: length numel(chord_vals) +
%                        numel(tmpl_vals) - 1 (non-periodic), or one
%                        value per grid point of the period (periodic).
%
%   See also TEMPLATEHARMONICITY, VIRTUALPITCHES.

    if nargin < 10 || isempty(per)
        per = false;
    end
    if nargin < 11 || isempty(period)
        period = 1200;
    end

    if per
        chord_dens = buildMaet(chord_p, chord_w, sigma, 1, false, ...
            true, period, 'verbose', false);
        x_chord = internal.periodicGrid(period, step);
    else
        chord_dens = buildMaet(chord_p, chord_w, sigma, 1, false, ...
            false, 1200, 'verbose', false);
        x_chord = -margin:step:(max(chord_p) + margin);
    end
    chord_vals = evalMaet(chord_dens, x_chord, ...
        'truncationSigmas', truncationSigmas, ...
        'kernelPrecision', kernelPrecision, ...
        'verbose', false);

    if per
        % Circular cross-correlation: lag l places the template's
        % fundamental at l * step.
        xcorr_vals = real(ifft(fft(chord_vals(:)) .* conj(fft(tmpl_vals(:)))));
    else
        xcorr_vals = conv(chord_vals, fliplr(tmpl_vals), 'full');
    end
    norm_factor = sqrt(sum(chord_vals .^ 2) * tmpl_norm_sq);
    xcorr_norm = xcorr_vals(:) / norm_factor;
end
