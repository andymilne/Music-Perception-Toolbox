function s = expTensorSim(x_p, x_w, y_p, y_w, sigma, kerLen, ...
                          r, isRel, isPer, limits,...
                          method, metric, p)
%EXPTENSORDIST Similarity between two pitch multisets as expectation tensors
%
%   s = expTensorSim(x_p, x_w, y_p, y_w, sigma, kerLen, r, isRel, isPer,
%   limits, method, metric, p)
%
%   Given two pitch multisets x_p and y_p with respective weights x_w and y_w,
%   this function gives the similarity of their r-ad expectation densities.
%
%   The returned value can be cosine similarity 'cosine' or the p-norm distance
%   'pDist' (set by the 'metric' argument). Note that the former is a
%   similarity measure (higher values mean more similar), the latter a distance
%   measure (higher values mean greater distance or less similarity). 
%
%   Two methods are available: 'numeric' and 'analytic'. Depending on the
%   inputs, one of these methods may be substantially faster than the other.
%   When the 'method' argument is empty, it is chosen automatically with a
%   crude heuristic that may choose the slower method. For this reason, when
%   multiple distances are being calculated (e.g., in a loop), it is best to
%   test 'numeric' and 'analytic' on a single example to see which is the
%   faster method.
%
%   For explanations of the remaining arguments, see expectationTensor.
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University
%
%   See also EXPECTATIONTENSOR, ANALYTICTENSOR.

persistent X Y x_pLast x_wLast y_pLast y_wLast sigmaLast kerLenLast ...
           rLast isRelLast isPerLast limitsLast methodLast metricLast pLast
       
if nargin < 13
    p = 2;
end
if nargin < 12
    metric = 'cosine';
end
% Crude heuristic for choosing method when unspecified in the arguments
if nargin<11 
    if numel(limits) == 1
        limits(2) = limits(1);
        limits(1) = 0;
    end
    nDimX = r-isRel;
    if nDimX==2 && numel(x_p)<10 && numel(y_p)<10
        method = 'analytic';
    elseif nDimX==2
        method = 'numeric';
    elseif nDimX==3 && numel(x_p)<24 && numel(y_p)<24
        method = 'analytic';
    elseif nDimX==3
        method = 'numeric';
    elseif nDimX > 3 
        method = 'analytic';
    end
end

% % Alternative heuristic 
% complexity = (kerLen*sigma)^(r-isPer) ...
%            * factorial(numel(x_p)) ...
%            / factorial(numel(x_p)-r+isRel)

if strcmpi(method,'analytic') || strcmpi(method,'ana')
    method = 'analytic';
elseif strcmp(method,'numeric') || strcmpi(method,'num')
    method = 'numeric';
end
if strcmpi(metric,'cosine') || strcmpi(metric,'cos')
    metric = 'cosine';
elseif strcmpi(metric,'pDist')
    metric = 'pDist';
end

% Test for whether X or Y need to be recalculated
if ~isequal(x_p,x_pLast) || ~isequal(x_w,x_wLast)
    XNew = 1;
else
    XNew = 0;
end
if ~isequal(y_p,y_pLast) || ~isequal(y_w,y_wLast)
    YNew = 1;
else
    YNew = 0;
end
if ~isequal(sigma,sigmaLast) || ~isequal(kerLen,kerLenLast) ...
        || ~isequal(r,rLast) || ~isequal(isRel,isRelLast) ...
        || ~isequal(isPer,isPerLast) || ~isequal(limits,limitsLast) ...
        || ~isequal(method,methodLast) || ~isequal(metric,metricLast) ...
        || ~isequal(p,pLast)
    XNew = 1;
    YNew = 1;
end

% Analytic cosine (calls David Bulger's cosSimExpTens function)
if strcmp(method,'analytic') && strcmp(metric,'cosine')
    s = cosSimExpTens(x_p,x_w,y_p,y_w,sigma,r,isRel,isPer,limits(end));
    
% Analytic pDist (not supported)
elseif strcmp(method,'analytic') && strcmp(metric,'pDist')
    error(['The analytic method does not support pDist. ' ... 
           'Change method or use cosine similarity.'])

% Numeric cosine (calls my expectationTensor function)
elseif strcmp(method,'numeric') && strcmp(metric,'cosine')
    if XNew
        X = expectationTensor(x_p,x_w,sigma,kerLen,r,isRel,isPer,limits);
    end
    if YNew
        Y = expectationTensor(y_p,y_w,sigma,kerLen,r,isRel,isPer,limits);
    end
    s = spCosSim(X,Y);

% Numeric pDist % Numeric cosine (calls my expectationTensor function)
elseif strcmp(method,'numeric') && strcmp(metric,'pDist')
    if XNew
        X = expectationTensor(x_p,x_w,sigma,kerLen,r,isRel,isPer,limits);
    end
    if YNew
        Y = expectationTensor(y_p,y_w,sigma,kerLen,r,isRel,isPer,limits);
    end
    s = spPDist(X,Y,p);
end

% Store last arguments
x_pLast = x_p;
x_wLast = x_w;
y_pLast = y_p;
y_wLast = y_w;
sigmaLast = sigma;
kerLenLast = kerLen;
rLast = r;
isRelLast = isRel;
isPerLast = isPer;
limitsLast = limits;
methodLast = method;
metricLast = metric;
pLast = p;

end