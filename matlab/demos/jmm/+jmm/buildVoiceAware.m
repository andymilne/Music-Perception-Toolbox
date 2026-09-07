function dens = buildVoiceAware(satbCents, sigmaPc, sigmaPh)
%BUILDVOICEAWARE  Analysis 1.2, encoding (i): the voice-aware chord density.
%
%   dens = jmm.buildVoiceAware(satbCents, sigmaPc, sigmaPh)
%
%   Two attributes (pitch class, pitch height), each the ordered
%   (S, A, T, B) voicing: K = 4 per event, [sym] = 0 (ordered), r = 4.
%   Numerically identical to four r = 1 per-voice attributes. Pitch class
%   is periodic (P = 1200 cents); pitch height is not. satbCents is the
%   chord's four pitches in cents, soprano first.
%
%   Twin of build_voice_aware in demo_jmm_1_2_similarity.py.
%
%   See also JMM.BUILDSIMPLEXVOICE, JMM.BUILDVOICEAGNOSTIC, BUILDEXPTENS.
    voicing = double(satbCents(:));                 % K = 4, N = 1
    pAttr = {voicing, voicing};
    dens = buildExpTens( ...
        pAttr, [], ...
        [sigmaPc, sigmaPh], ...      % sigma per attribute
        [4, 4], ...                  % r per attribute
        [false, false], ...          % isRel
        [true, false], ...           % isPer: PC periodic, pitch height not
        [1200.0, 0.0], ...           % period
        [false, false], ...          % isSym = 0 -> ordered (voice-aware)
        'verbose', false);
end
