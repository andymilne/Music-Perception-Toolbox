function r = fSetRoughness(x_f,x_w,pNorm,isAve)
%FSETROUGHNESS Roughness of partials with frequencies 'x_f' and weights 'x_w'.
%
%   r = fSetRoughness(x_f, x_w, pNorm, isAve): The total roughness of the set
%   of partials with frequencies 'x_f' and weights 'x_w'.
%
%   pNorm sets the norm by which the total roughness is calculated from the
%   "subroughnesses" of each partial pair. It defaults to 1, which is simple
%   summation.
%
%   isAve = 0 or 1 - when 1, the total roughness is divided by the total number
%   of pairs of partials. When pNorm is 1, this gives the expected roughness of
%   a partial pair. The default is 0.
%
%   This function uses Bill Sethares' (1993) parameterization of Plomp-Levelt's
%   (1965) empirical dissonance curve.
%
%   Plomp, R. and Levelt, W. J. M. (1965). Tonal consonance and critical
%   bandwidth. The Journal of the Acoustical Society of America, 38(4):548-560.
%
%   Sethares, W. A. (1993). Local consonance and the relationship between
%   timbre and scale. The Journal of the Acoustical Society of America,
%   94(3):1218-1228.
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University
%
%   See also: pSetSpectralEntropy

if nargin < 4
    isAve = 0;
end
if nargin < 3
    pNorm = 1;
end

if pNorm <= 0
    error('pNorm must be a positive number')
end
if ~isequal(isAve,0) && ~isequal(isAve,1)
    error('isAve must be 0 or 1')
end

if isempty(x_w)
    x_w = ones(numel(x_p),1);
end
if numel(x_w) == 1
    if x_w == 0
        warning('All weights in x_w are zero.');
    end
    x_w = x_w*ones(numel(x_p),1);
end
x_w = x_w(:);

%% Fixed parameters
Dstar = 0.24;
S1 = 0.0207;
S2 = 18.96;
C1 = 5;
C2 = -5;
A1 = -3.51;
A2 = -5.75;

%% Variables
x_f = x_f(:);
x_w = x_w(:);

nPartials = numel(x_f);

fDiff = x_f - x_f';
fDiff = fDiff(:);

fMin = min(x_f,x_f');
fMin = fMin(:);

aMin = min(x_w,x_w');
aMin = aMin(:);

validInd = fDiff>0;
fMin = fMin(validInd);
aMin = aMin(validInd);
fDiff = fDiff(validInd);

fMid = fMin + fDiff/2; % This may be useful when attempting to take into
% account modifications suggested in Parncutt (2006) and Vos (2006)
% commentaries on Mashinter (2006) regarding summations within and between CBs.

%% Calculate subroughnesses
allS = Dstar./(S1*fMin + S2);
allRough = aMin.*(C1*exp(A1*allS.*fDiff) + C2*exp(A2*allS.*fDiff));

%% Calculate total roughness
r = sum(allRough.^pNorm)^(1/pNorm);
if isAve == 1
    r = r/nchoosek(nPartials,2);
end

end
