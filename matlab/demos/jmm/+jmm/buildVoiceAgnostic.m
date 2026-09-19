function dens = buildVoiceAgnostic(satbCents, sigmaPc, sigmaPh)
%BUILDVOICEAGNOSTIC  Analysis 1.2, encoding (iii): the voice-agnostic density.
%
%   dens = jmm.buildVoiceAgnostic(satbCents, sigmaPc, sigmaPh)
%
%   Two attributes (pitch class, pitch height), one event per note
%   (N = 4, K = 1, r = 1): the simplex-voice encoding without its voice
%   attribute. Each event binds a note's pitch class to its own pitch
%   height; a single K = 4 event on both attributes would instead tensor
%   the pitch class of one note with the height of another (16 cross
%   terms) and lose that binding. Voice identity is not encoded.
%   satbCents is the chord's four pitches in cents.
%
%   Twin of build_voice_agnostic in demo_jmm_1_2_similarity.py.
%
%   See also JMM.BUILDVOICEAWARE, JMM.BUILDSIMPLEXVOICE, BUILDMAET.
    p4 = double(satbCents(:)).';                    % K = 1, N = 4
    dens = buildMaet( ...
        {p4, p4}, [], ...
        [sigmaPc, sigmaPh], ...
        [1, 1], ...
        [false, false], ...
        [true, false], ...
        [1200.0, 0.0], ...
        [true, true], ...            % moot at K = 1, r = 1
        'verbose', false);
end
