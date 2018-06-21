function s = contextProbeSpecSim(contextPcs,contextWts,...
                                 probePcs,probeWts,...
                                 sigma,rho,nHarm,method)
%CONTEXTPROBESPECSIM Spectral similarities of probes with context
%   Given the vector contextPcs comprising N context pcs (in cents), the vector
%   contextWts comprising their associated nonnegative weights, a JxK matrix
%   probesPcs comprising J probe pcs (in cents) in each of K probe pc sets, and
%   their associated weights in the matrix probesWts,
%
%   function s = contextProbeSpecSim(contextPcs, contextWts,...
%                                    probesPcs, probesWts, ...
%                                    sigma, rho, numHarm)
%
%   returns the K spectral pitch similarities between the K probe sets and the
%   context. Every pc is spectralized wih harmonic partials h = (1, 2, ...,
%   nHarm) with weights h^-rho times the associated context or probe weight.
%   Sigma sets the smoothing width for the spectral pitch similarity
%   calculation. If contextWts or probesWts is a scalar, all associated 
%   weights are set to that scalar; if contextWts or probesWts is empty or 0,
%   all associated weights are set to 1.
%
%   method = 'numeric' or 'analytic' -- in this context, the latter is slightly
%   more accurate but slower.

if nargin < 8
    method = 'numeric';
end
if nargin < 7
    nHarm = 12;
end
if nargin < 6
    rho = 0.67;
end
if nargin < 5
    sigma = 6;
end

nContextWts = numel(contextWts);
if nContextWts <= 1
    if isempty(contextWts) || isequal(contextWts,0)
        contextWts = ones(size(contextPcs));
    else
        contextWts = probeWts*ones(size(contextPcs));
    end
end

nProbes = size(probePcs,1);
nProbeSets = size(probePcs,2);
s = nan(nProbeSets,1);

nProbesWts = numel(probeWts);
if nProbesWts <= 1
    if isempty(probeWts) || isequal(probeWts,0)
        probeWts = ones(size(probePcs));
    else
        probeWts = probeWts*ones(size(probePcs));
    end
end

%% Fixed parameters
q = 1200;
kerLen = 9;

%% Add harmonics with spectralize function
harmNos = 1:nHarm;
specPc = 1200*log2(harmNos);
specWt = harmNos.^-rho;

[~,contextSpecPc,contextSpecWt] ...
    = spectralize(contextPcs,contextWts,specPc,specWt);
contextSpecPc = contextSpecPc(:);
contextSpecWt = contextSpecWt(:);

probesSpecPcMat = nan(nProbes*nHarm,nProbeSets);
probesSpecWtMat = probesSpecPcMat;
for i = 1:nProbeSets
    [~,probesSpecPc,probesSpecWt] ...
        = spectralize(probePcs(:,i),probeWts(:,i),specPc,specWt);
    probesSpecPcMat(:,i) = probesSpecPc(:);
    probesSpecWtMat(:,i) = probesSpecWt(:);
end

%% Here, I could add nonlinear distortion to each spectralized probe set

%% Scale-probe spectral similarities
x_p = contextSpecPc;
x_w = contextSpecWt;
for i = 1:nProbeSets
    y_p = probesSpecPcMat(:,i);
    y_w = probesSpecWtMat(:,i);
    r = 1; % monad expectation tensor
    isRel = 0; % tensor is not tranpositionally invariant
    isPer = 1; % periodic over limits
    limits = q;
    s(i) = expTensorSim(x_p, x_w, y_p, y_w, ...
                        sigma, kerLen, ...
                        r, isRel, isPer, limits, method);
end

end
