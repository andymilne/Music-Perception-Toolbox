%MARKOVS Optimal S-step Markov predictor for a periodic indicator sequence.
%
%   y = MarkovS(x_ind, S): x_ind is the pitch/time class indicator; S is
%   the length of sequences considered.
% 
%   By David Bulger, Macquarie University.

function y = MarkovS(x_ind, S)

if nargin<2
    S = 3;
end

x_ind = x_ind(:)';
N = length(x_ind);
E = x_ind==x_ind'; 
T = true(N);
for k = 1:S
    T = T & circshift(circshift(E,k,2),k,1);
end

y = (x_ind*double(T)) ./ sum(T);
