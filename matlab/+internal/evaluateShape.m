function h = evaluateShape(delta, sd, gamma)
%LOCALEVALUATESHAPE  Peak-normalised rect * Gaussian convolution
%(Section 5.2.1 of the MAET manuscript), with derived parameters
%   phi = sd * sqrt(3 * gamma)   (rectangle half-width)
%   xi  = sd * sqrt(1 - gamma)   (Gaussian std)
%so that the total variance is sd^2 across the whole family.
    if gamma == 0
        % Pure Gaussian, std = sd.
        h = exp(-(delta .^ 2) ./ (2 * sd ^ 2));
        return;
    end
    if gamma == 1
        % Pure rectangle, half-width phi = sd * sqrt(3). Half-open support
        % [-phi, phi): lower edge included, upper edge excluded, so a
        % regular pulse grid yields exactly N pulses for full support
        % N*IOI at every N (a closed interval over-counts even widths and
        % can leave a between-pulse centre empty). The tolerance keeps the
        % edge test robust to floating-point error.
        phi = sd * sqrt(3);
        scale = max(abs(phi), 1);
        if ~isempty(delta)
            scale = max(scale, max(abs(delta(:))));
        end
        tol = 1e-9 * scale;
        h = double((delta >= -phi - tol) & (delta < phi - tol));
        return;
    end
    phi   = sd * sqrt(3 * gamma);
    xi    = sd * sqrt(1 - gamma);
    scale = xi * sqrt(2);
    num   = erf((delta + phi) ./ scale) - erf((delta - phi) ./ scale);
    peak  = 2 * erf(phi ./ scale);
    h = num ./ peak;
end
