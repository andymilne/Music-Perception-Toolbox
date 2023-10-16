%% TonesSpectralSim
% This routine calculates spectral pitch similarity of a fixed "reference" 
% tone and a smoothly varying "continuum" tone.

% Author: Andrew Milne 

% Reference: Milne, A.J., Sethares, W.A., Laney, R., Sharp, D.B. (2011)
% Modelling the Similarity of Pitch Collections with Expectation Tensors,
% Journal of Mathematics and Music, 5(1), 1-20.

%% Parameters
% See expectationTensor for explanations of these parameters
sigma = 12;
kerLen = 6;
r = 1;
isRel = 0;
isPer = 1;
limits = 1200;
method = 'analytic';

%% Reference and generated triad parameters
% "low" and "high" sets the range (in cents) of the continuum tone relative to 
% the reference tone.
%
% "inc" sets the size, in cents, of the increments between "low" and
% "high".
%
% "num_harmonics" sets the number of spectral pitches in each tone.
%
% "rho" sets the amplitude of each tone's harmonics to 1/n^rho, where n is
% the harmonic number; rho = 0 gives a flat spectrum (all magnitudes are 1;
% rolloff = 1 gives a sawtooth spectrum; rolloff = a very big number
% results in all harmonics, except the fundamental, having almost zero
% magnitude.
%
% See expTensorSim for meanings of the other arguments.

low = 0;
high = 1200;
inc = 1;
num_harmonics = 32;
rho = 1;

%% Generate reference tone's spectrum
x_p = zeros(1, num_harmonics);
for i = 1:num_harmonics
    x_p(i) = 1200*log2(i);
end

x_w = zeros(1, num_harmonics);
for i = 1:num_harmonics
    x_w(i) = i^(-rho);
end

%% Generate continuum tones
y_w = x_w;
shifts = low:inc:high;
Dist = zeros(length(shifts),length(shifts));

sim_xy = zeros(1, limits);
for i = 1:length(shifts)
    y_p(1,:) = x_p(1,:)+shifts(i);
    sim_xy(i) = expTensorSim(x_p, x_w, y_p, y_w, ...
        sigma, kerLen, r, isRel, isPer, limits, method);
end

%% Plotting
figure(1)
stairs(sim_xy)
xlim([0 limits])
