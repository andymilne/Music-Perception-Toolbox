%PROJCENT The signed magnitude of the projected centroid at each unit of x_pc.
%
%   y = projCent(x_pc, period): y is the 
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University.

function y = projCent(x_pc, period)

x_pc = x_pc(:)';
N = period;
K = numel(x_pc);

pattern_circle = exp(2*pi*1i*(x_pc)/N);
pattern_circle2 = exp(2*pi*1i*(x_pc)/N)

DFT_circ = fft(pattern_circle)/K;

% This gives the angle of the densest point of the circle.
bal_phase = mod(angle(DFT_circ(0+1)),2*pi);

% This gives the magnitude of the imbalance.
bal_mag = abs(DFT_circ(0+1));

% The multiplication by the cosine of the angle between bal_phase and the
% pulses, projects the balance vector onto each pulse/chroma angle.
y = (bal_mag.*cos(bal_phase - 2*pi*(0:N-1)'/N))'; 