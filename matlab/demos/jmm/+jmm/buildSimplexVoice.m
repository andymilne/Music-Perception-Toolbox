function dens = buildSimplexVoice(satbCents, sigmaPc, sigmaPh, sigmaVoice)
%BUILDSIMPLEXVOICE  Analysis 1.2, encoding (ii): the simplex-voice density.
%
%   dens = jmm.buildSimplexVoice(satbCents, sigmaPc, sigmaPh)
%   dens = jmm.buildSimplexVoice(satbCents, sigmaPc, sigmaPh, sigmaVoice)
%
%   One event per voice (N = 4): pitch-class and pitch-height attributes
%   (r = 1) plus a voice attribute carrying the simplex vertex
%   (simplexVertices(4)) as an ordered categorical attribute (K = 3
%   coordinates, [sym] = 0, r = 3, sigmaVoice default 0.2). satbCents is
%   the chord's four pitches in cents, soprano first.
%
%   Twin of build_simplex_voice in demo_jmm_1_2_similarity.py.
%
%   See also JMM.BUILDVOICEAWARE, JMM.BUILDVOICEAGNOSTIC, SIMPLEXVERTICES.
    if nargin < 4 || isempty(sigmaVoice), sigmaVoice = 0.2; end
    pitches = double(satbCents(:)).';               % 1 x 4: K = 1, N = 4
    voice = simplexVertices(4).';                   % 3 x 4: 3 coords x 4 voices
    pAttr = {pitches, pitches, voice};
    dens = buildExpTens( ...
        pAttr, [], ...
        [sigmaPc, sigmaPh, sigmaVoice], ...
        [1, 1, 3], ...
        [false, false, false], ...
        [true, false, false], ...
        [1200.0, 0.0, 0.0], ...
        [true, true, false], ...     % voice attribute ordered ([sym] = 0)
        'verbose', false);
end
