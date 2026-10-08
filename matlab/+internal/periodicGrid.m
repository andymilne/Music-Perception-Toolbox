function x = periodicGrid(period, step)
%PERIODICGRID  Grid 0:step:(period - step) on one period.
%
%   x = internal.periodicGrid(period, step)
%
%   step must divide period (to a relative tolerance of 1e-9), so that
%   the grid closes on itself and circular operations on it are exact.
%   Returns a row vector of round(period / step) points, the same
%   points entropyMaet uses on a periodic attribute with that many
%   points per dimension.
%
%   See also INTERNAL.TEMPLATEVALUES, SPECTRALENTROPY.

    if ~(period > 0)
        error('internal:periodicGrid:badPeriod', 'period must be positive.');
    end
    n = round(period / step);
    if n < 2 || abs(n * step - period) > 1e-9 * period
        error('internal:periodicGrid:stepDoesNotDividePeriod', ...
            'resolution (%g) must divide the period (%g) when ''per'' is true.', ...
            step, period);
    end
    x = (0:n-1) * step;
end
