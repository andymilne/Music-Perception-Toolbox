function sv = sweepValues(plan, A)
%SWEEPVALUES  The sweep values of each swept attribute, from a sweep plan.
%
%   sv = internal.sweepValues(plan, A) returns a 1 x A cell: sv{a} holds
%   attribute a's sweep values, listed or generated, and is empty where
%   attribute a is not swept. Under 'independent', sv{a} is the pair
%   {windowValues, queryValues}. They are the axes of the swept
%   functions' output, one per dimension. Twin of the Python
%   _sweep_values.
%
%   See also internal.sweptPlan, SWEPTSIMILARITY, SWEPTENTROPY, SWEPTMASS.

sv = cell(1, A);
for d = plan.dims
    if d.pair > 0
        sv{d.a} = {sv{d.a}, d.vals};
    else
        sv{d.a} = d.vals;
    end
end
end
