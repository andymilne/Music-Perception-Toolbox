function tf = spectralIpEnabled(value)
%INTERNAL.SPECTRALIPENABLED  Master switch for the spectral IP branch.
%
%   TF = INTERNAL.SPECTRALIPENABLED() returns whether the spectral
%   (Fourier) branch of the relative-mode inner product is enabled.
%   INTERNAL.SPECTRALIPENABLED(VALUE) sets it and returns the new state.
%
%   Exists so tests can force the translation-grid route and compare the
%   two, exactly as the Python tests toggle
%   cosine._SPECTRAL_IP_ENABLED. Both routes compute the full-image
%   measure, so the switch changes cost, not value.
%
%   Mirror of Python cosine._SPECTRAL_IP_ENABLED.
    persistent enabled
    if isempty(enabled)
        enabled = true;
    end
    if nargin > 0
        enabled = logical(value);
    end
    tf = enabled;
end
