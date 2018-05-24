function s = contextProbeSpecSim(contextPcs,contextWts,...
                                 probesPcs,probesWts,...
                                 sigma,rho,nHarm)
%CONTEXTPROBESPECSIM Spectral similarities of probes with context
%   Given a vector of context pcs (in cents) and its associated weights vector,
%   and an MxN matrix of probe pcs, where each column is a different set of N
%   probes and each of the M rows is a pc in that set (in cents)
%
%   function s = contextProbeSpecSim(contextPcs, contextWts,...
%                                    probesPcs, probesWts, ...
%                                    sigma, rho, numHarm)
%
%   returns the N spectral pitch similarities between the N probe sets and the
%   context. Every pc is spectralized wih harmonic partials h = (1, 2, ...,
%   nHarm) with weights h^-rho times the associated context or probe weight.
%   Sigma sets the smoothing width for the spectral pitch similarity
%   calculation. If contextWts or probesWts is 1 or empty, all weights are set
%   to 1.

if nargin < 7
    nHarm = 12;
end
if nargin < 6
    rho = 0.67;
end
if nargin < 5
    sigma = 6;
end

nContextPcs = numel(contextPcs);
nContextWts = numel(contextWts);
if nContextWts == 1
    contextWts = probesWts*ones(nContextPcs,1);
end

nProbes = size(probesPcs,2);
s = nan(nProbes,1);

nProbesPcs = numel(probesPcs);
nProbesWts = numel(probesWts);
if numel(probesWts) == 1
    probesWts = probesWts*ones(size(probesPcs));
elseif nProbesWts == 1
    probesWts = ones(nProbesPcs,1).*probesWts;
end

%% Fixed parameters
q = 1200;
kerLen = 9;

%% Add harmonics
harmNos = 1:nHarm;
specPc = 1200*log2(harmNos);
specWt = harmNos.^-rho;

[~,contextSpecPc,contextSpecWt] ...
    = spectralize(contextPcs,contextWts,specPc,specWt);
contextSpecPc = contextSpecPc(:);
contextSpecWt = contextSpecWt(:);

probesSpecPcMat = nan(size(probesPcs,1)*nHarm,nProbes);
probesSpecWtMat = probesSpecPcMat;
for i = 1:nProbes
    [~,probesSpecPc,probesSpecWt] ...
        = spectralize(probesPcs(:,i),probesWts(:,i),specPc,specWt);
    probesSpecPcMat(:,i) = probesSpecPc(:);
    probesSpecWtMat(:,i) = probesSpecWt(:);
end

%% Here, I could add nonlinear distortion to each spectralized probe set

%% Scale-probe spectral similarities
x_p = contextSpecPc;
x_w = contextSpecWt;
for i = 1:nProbes
    y_p = probesSpecPcMat(:,i);
    y_w = probesSpecWtMat(:,i);
    r = 1; % monad expectation tensor
    isRel = 0; % tensor is not tranpositionally invariant
    isPer = 1; % periodic over limits
    limits = q;
    s(i) = expTensorSim(x_p, x_w, y_p, y_w, ...
                        sigma, kerLen, ...
                        r, isRel, isPer, limits);
end

end
