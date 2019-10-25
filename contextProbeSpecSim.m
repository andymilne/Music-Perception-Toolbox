function s = contextProbeSpecSim(context_pc,context_w,...
                                 probe_pc,probe_w,...
                                 sigma,rho,nHarm,method)
%CONTEXTPROBESPECSIM Spectral similarities of probes with context
%   Given the vector context_pc comprising N context pcs (in cents), the vector
%   context_w comprising their associated nonnegative weights, a JxK matrix
%   probesPcs comprising J probe pcs (in cents) in each of K probe pc sets, and
%   their associated weights in the matrix probesWts,
%
%   function s = contextProbeSpecSim(context_pc, context_w,...
%                                    probesPcs, probesWts, ...
%                                    sigma, rho, numHarm)
%
%   returns the K spectral pitch similarities between the K probe sets and the
%   context. Every pc is spectralized wih harmonic partials h = (1, 2, ...,
%   nHarm) with weights h^-rho times the associated context or probe weight.
%   Sigma sets the smoothing width for the spectral pitch similarity
%   calculation. If context_w or probesWts is a scalar, all associated 
%   weights are set to that scalar; if context_w or probesWts is empty or 0,
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

ncontext_w = numel(context_w);
if ncontext_w <= 1
    if isempty(context_w) || isequal(context_w,0)
        context_w = ones(size(context_pc));
    else
        context_w = probe_w*ones(size(context_pc));
    end
end

nProbes = size(probe_pc,1);
nProbeSets = size(probe_pc,2);
s = nan(nProbeSets,1);

nProbesWts = numel(probe_w);
if nProbesWts <= 1
    if isempty(probe_w) || isequal(probe_w,0)
        probe_w = ones(size(probe_pc));
    else
        probe_w = probe_w*ones(size(probe_pc));
    end
end

%% Fixed parameters
limits = 1200;
kerLen = 9;

%% Add harmonics with spectralize function
harmNos = 1:nHarm;
specPc = 1200*log2(harmNos);
specWt = harmNos.^-rho;

[~,contextSpecPc,contextSpecWt] ...
    = spectralize(context_pc,context_w,specPc,specWt);
contextSpecPc = contextSpecPc(:);
contextSpecWt = contextSpecWt(:);

probesSpecPcMat = nan(nProbes*nHarm,nProbeSets);
probesSpecWtMat = probesSpecPcMat;
for i = 1:nProbeSets
    [~,probesSpecPc,probesSpecWt] ...
        = spectralize(probe_pc(:,i),probe_w(:,i),specPc,specWt);
    probesSpecPcMat(:,i) = probesSpecPc(:);
    probesSpecWtMat(:,i) = probesSpecWt(:);
end

%% Scale-probe spectral similarities
x_p = contextSpecPc;
x_w = contextSpecWt;
for i = 1:nProbeSets
    y_p = probesSpecPcMat(:,i);
    y_w = probesSpecWtMat(:,i);
    r = 1; % monad expectation tensor
    isRel = 0; % tensor is not tranpositionally invariant
    isPer = 1; % periodic over limits
    s(i) = expTensorSim(x_p, x_w, y_p, y_w, ...
                        sigma, kerLen, ...
                        r, isRel, isPer, limits, method);
end

end
