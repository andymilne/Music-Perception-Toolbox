function d = expTensorDist(x_p, x_w, y_p, y_w, sigma, kerLen, ...
                           r, isRel, isPer, limits,...
                           method, metric, p)
%EXPTENSORDIST  Distance between two pitch multisets as expectation tensors
%   Given two pitch multisets x_p and y_p with respective weights x_w and y_w,
%   this function gives the distance between their r-ad expectation densities.
%
%   The distance can be cosine similarity or pDist (set by the 'metric'
%   argument).
%
%   Two methods are available: numeric and analytic. For r > 2, period = 1200,
%   sigma and kerLen both about 6, and number of pitches < 10, analytic is
%   typically faster. When the method argument is empty, it is chosen
%   automatically with a very approximate heuristic that is likely to choose
%   the slower method. For this reason, when multiple distances are being
%   calculated (e.g., in a loop), it is best to test numeric and analytic on a
%   single example to see which is the faster method.

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

% Analytic cosine
if strcmp(method,'analytic') && strcmp(metric,'cosine')
    d = cosSimExpTens(x_p,x_w,y_p,y_w,sigma,r,isRel,isPer,limits(end));
    
% Analytic pDist (not supported)
elseif strcmp(method,'analytic') && strcmp(metric,'pDist')
    error(['The analytic method does not support pDist. ' ... 
           'Change method or use cosine distance.'])

% Numeric cosine
elseif strcmp(method,'numeric') && strcmp(metric,'cosine')
    if XNew
        X = expectationTensor(x_p,x_w,sigma,kerLen,r,isRel,isPer,limits);
    end
    if YNew
        Y = expectationTensor(y_p,y_w,sigma,kerLen,r,isRel,isPer,limits);
    end
    d = spCosSim(X,Y);

% Numeric pDist
elseif strcmp(method,'numeric') && strcmp(metric,'pDist')
    if XNew
        X = expectationTensor(x_p,x_w,sigma,kerLen,r,isRel,isPer,limits);
    end
    if YNew
        Y = expectationTensor(y_p,y_w,sigma,kerLen,r,isRel,isPer,limits);
    end
    d = spPDist(X,Y,p);
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