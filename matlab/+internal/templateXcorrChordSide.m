function xcorr_norm = templateXcorrChordSide( ...
        chord_p, chord_w, sigma, ...
        tmpl_vals, tmpl_norm_sq, margin, step, ...
        truncationSigmas, kernelPrecision)
%TEMPLATEXCORRCHORDSIDE  Normalised cross-correlation profile of a chord
%against a pre-evaluated template spectrum.
%
%   xcorr_norm = internal.templateXcorrChordSide( ...
%       chord_p, chord_w, sigma, ...
%       tmpl_vals, tmpl_norm_sq, margin, step, ...
%       truncationSigmas, kernelPrecision)
%
%   Builds a 1-D absolute non-periodic expectation tensor from
%   (chord_p, chord_w) under Gaussian smoothing sigma, evaluates it
%   on the grid 0:step:(max(chord_p) + margin), convolves with the
%   reverse of the supplied template values, and normalises by
%   sqrt(sum(chord_vals.^2) * tmpl_norm_sq) so each lag is a cosine
%   similarity in [0, 1].
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
%     margin           - Grid margin in cents (typically 4 * sigma).
%     step             - Grid spacing in cents (typically 1).
%     truncationSigmas - Forwarded to evalExpTens.
%     kernelPrecision  - Forwarded to evalExpTens.
%
%   Output:
%     xcorr_norm       - Column vector of normalised cross-correlation
%                        values, length numel(chord_vals) + numel(tmpl_vals) - 1.
%
%   See also TEMPLATEHARMONICITY, VIRTUALPITCHES.

    chord_dens = buildExpTens(chord_p, chord_w, sigma, 1, false, ...
        false, 1200, 'verbose', false);
    x_chord = 0:step:(max(chord_p) + margin);
    chord_vals = evalExpTens(chord_dens, x_chord, ...
        'truncationSigmas', truncationSigmas, ...
        'kernelPrecision', kernelPrecision, ...
        'verbose', false);

    xcorr_vals = conv(chord_vals, fliplr(tmpl_vals), 'full');
    norm_factor = sqrt(sum(chord_vals .^ 2) * tmpl_norm_sq);
    xcorr_norm = xcorr_vals(:) / norm_factor;
end
