function e = evennessCircular(p, period)
%EVENNESSCIRCULAR Evenness of a circular multiset.
%
%   e = evennessCircular(p, period):
%
%   Computes the evenness of a multiset of K points on a circle
%   (p represents pitches or positions), defined as:
%
%     e = |F(1)|
%
%   where F(k) is the k-th DFT coefficient of the multiset
%   (see dftCircular). The k = 1 coefficient captures the extent
%   to which the K sorted elements match a maximally even
%   (equal-step) distribution around the circle. For a maximally
%   even multiset, each sorted element j (0-indexed) is at position
%   approximately j * period / K, so:
%     z(j) * exp(-2*pi*1i*j/K) = exp(2*pi*1i*j/K) * exp(-2*pi*1i*j/K) = 1
%   and |F(1)| = 1.
%
%   Evenness ranges from 0 to 1:
%     e = 1: maximally even — the multiset consists of K equally spaced
%            Examples: the whole-tone scale (6 notes in 12-EDO), the
%            chromatic scale, an isochronous rhythm.
%     e = 0: maximally uneven for this cardinality.
%
%   Maximal evenness implies perfect balance, but perfect balance does not
%   imply maximal evenness: a multiset can be perfectly balanced without
%   being maximally even (see balanceCircular).
%
%   Evenness always uses uniform (binary) weights, following Milne et al.
%   (2017): "we focus on binary-weighted patterns, whose weights are all
%   zero or one." Evenness is a property of the spatial distribution of
%   elements around the circle, not of their relative saliences. See
%   balanceCircular for a measure that supports non-uniform weights.
%
%   For further information, see:
%     Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the
%       space of perfectly balanced rhythms and scales. Journal of
%       Mathematics and Music, 11(2-3), 101-133.
%     Milne, A. J. & Herff, S. A. (2020). The perceptual relevance of
%       balance, evenness, and entropy in musical rhythms. Cognition,
%       203, 104233.
%
%   Inputs:
%     p      — Pitch or position values (vector of length K).
%     period — Period of the circular domain.
%
%   Output:
%     e      — Evenness (scalar, range [0, 1]).
%
%   Examples:
%     % Maximally even: whole-tone scale (6 equally spaced pitches)
%     e = evennessCircular([0, 200, 400, 600, 800, 1000], 1200);  % e = 1.000
%
%     % Nearly even: diatonic scale (7 notes, not equally spaced)
%     e = evennessCircular([0, 200, 400, 500, 700, 900, 1100], 1200);
%
%     % Rhythmic pattern: 5 onsets in a 16-step cycle
%     e = evennessCircular([0, 3, 6, 10, 13], 16);
%
%   See also balanceCircular, dftCircular.

[~, mag] = dftCircular(p, [], period);  % uniform weights
e = mag(2);  % mag(2) = |F(k=1)|

end
