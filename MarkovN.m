%MARKOVN Optimal N-step Markov predictor for the periodic indicator sequence.
%
%   y = MarkovN(x_ind, N): x_ind is the pitch/time class indicator; N are
%   the length of sequences considered. The output vector has the 
% 
%   By David Bulger, Macquarie University.

function y = MarkovN(x_ind, N)

if nargin<2
    N = 3;
end

x_ind = x_ind(:)';
N = length(x_ind);
E = x_ind==x_ind'; 
T = true(N);
for k = 1:N
    T = T & circshift(circshift(E,k,2),k,1);
end

y = (x_ind*double(T)) ./ sum(T);