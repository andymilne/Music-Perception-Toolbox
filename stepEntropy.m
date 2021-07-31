function h = stepEntropy(x_pc,period,k)
% STEPENTROPY  Normalized entropy of sequences of pitch/time step sizes.
% 
% Given a multiset x_pc of pitches/time classes,
%
% h = STEPENTROPY(x_pc,period,k)
%
% returns the normalized entropy of sequences of k - 1 step sizes between k
% pitches/events ordered by size. If k is omitted, it defaults to 2; if period
% is omitted, it defaults to 12.
% 
% For example, when considering k = 2, the diatonic pitch/time class set (0, 2,
% 4, 5, 7, 9, 11) has five steps of 2 semitones and two steps of 1 semitone.
% This defines a probability mass function over all 12 step sizes and, hence, a
% respective entropy value. When considering k = 3, the same indicator vector
% has three 2-tuples of sizes (2,2), two 2-tuples of sizes (2,1), two 2-tuples
% of sizes (1,2), and zero 2-tuples of sizes (1,1). This defines a probability
% mass function over all such tuples and, hence a respective entropy value.
%
% This function requires histcn: Bruno Luong (2021).
% N-dimensional histogram
% (https://www.mathworks.com/matlabcentral/fileexchange/23897-n-dimensional-histogram),
% MATLAB Central File Exchange. Retrieved May 1, 2021.
% 
% See also RADENTROPY and HISTENTROPY.

% Input checks
if nargin < 3
    k = 2;
end
if nargin < 2
    period = 12;
end

if exist('histcn') == 0
    error("The stepEntropy function requires the function histcn: https://www.mathworks.com/matlabcentral/fileexchange/23897-n-dimensional-histogram")
end

N = period;
x_pc = x_pc(:);
x_pc = sort(x_pc); % We are here interested in pitches/times ordered by size 
% (i.e., so seconds/IOIs are between  )
K = numel(x_pc);
eventIndex = x_pc + 1;

% Calculate all steps sizes
allRotations = eventIndex(mod((0 : K-1)' + (0 : K-1),K) + 1); % adapted from 
% https://au.mathworks.com/matlabcentral/fileexchange/22858-circulant-matrix/
% content/circulant.m
allStepSizes = mod(diff(allRotations,1),N);

% Entropy of consecutive events and permutations thereof
edges = (0:N) - 0.5;

% Calculate histograms and their entropies
if k<2 || k>8 || floor(k)~=k
    error('k must be an integer from 2 to 8.')
elseif k > K
    error(['k must be smaller than the number of notes/events (ones) in ' ...
           'the indicator vector.'])
elseif k == 2
    stepCounts = histcn(allStepSizes(1,:)',edges);
    h = histEntropy(stepCounts);
elseif k == 3
    stepCounts = histcn(allStepSizes(1:2,:)',edges,edges);
    h = histEntropy(stepCounts(:));
elseif k == 4
    stepCounts = histcn(allStepSizes(1:3,:)',edges,edges,edges);
    h = histEntropy(stepCounts(:));
elseif k == 5
    stepCounts = histcn(allStepSizes(1:4,:)',edges,edges,edges,edges);
    h = histEntropy(stepCounts(:));
elseif k == 6
    stepCounts = histcn(allStepSizes(1:5,:)',edges,edges,edges,edges,edges);
    h = histEntropy(stepCounts(:));
elseif k == 7
    stepCounts = histcn(allStepSizes(1:6,:)',edges,edges,edges,edges,edges,...
        edges);
    h = histEntropy(stepCounts(:));
elseif k == 8
    stepCounts = histcn(allStepSizes(1:7,:)',edges,edges,edges,edges,edges,...
        edges,edges);
    h = histEntropy(stepCounts(:));
end

end

