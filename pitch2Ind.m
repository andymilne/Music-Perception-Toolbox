function x_ind = pitch2Ind(x_p,x_w,period)
% Return the indicator vector for a pitch/time set

if nargin < 3
    period = 1200;
end
nPitches = numel(x_p);
if nargin == 2
    if isempty(x_w)
        x_w = 1;
    end
    if isscalar(x_w)
        x_w = x_w*ones(nPitches,1);
    end
end
if nargin<2
    x_w = ones(nPitches,1);
end

x_p = x_p(:);
x_p = round(x_p);

if numel(unique(x_p)) ~= nPitches
    warning('Pitches/times have been accumulated.')
end
subs = x_p+1;
x_ind = accumarray(subs,x_w,[period,1]);

end