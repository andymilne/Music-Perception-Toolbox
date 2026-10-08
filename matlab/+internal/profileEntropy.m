function h = profileEntropy(profile, method, base, step)
%PROFILEENTROPY  Entropy of a cross-correlation profile.
%
%   h = internal.profileEntropy(profile, method, base, step)
%
%   The profile is treated as a distribution over transpositions.
%   'shannon': H = -sum q log_b q of the profile normalized to unit
%   sum. 'normalized': H / log_b N, N the number of transpositions in
%   the profile. 'differential': H + log_b(step), the differential
%   entropy of the profile as a density over transposition in cents
%   (in units of log_b cents); independent of the grid spacing and of
%   how far the profile extends beyond its support.
%
%   See also TEMPLATEHARMONICITY.

    q = profile(:);
    N = numel(q);
    total = sum(q);
    if total > 0
        q = q / total;
    end
    q = q(q > 0);
    h = -sum(q .* log(q)) / log(base);
    switch method
        case 'normalized'
            h = h / (log(N) / log(base));
        case 'differential'
            h = h + log(step) / log(base);
    end
end
