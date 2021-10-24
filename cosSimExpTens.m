function s = cosSimExpTens(x_p,x_w,y_p,y_w,sigma,r,isRel,isPer,period)
%COSSIMEXPTENS Cosine similarity of two r-ad expectation tensors
%
%   s = cosSimExpTens(x_p, x_w, y_p, y_w, sigma, r, isRel, isPer, period):
%   Cosine similarity of two r-ad expectation tensors. The tensors can be
%   smoothed or unsmoothed, periodic or nonperiodic, absolute or relative.
%
%   Note that this function is usually faster than expectationTensor followed
%   by cosSim or spCosSim in cases where r > 2 and I < 10.
%
%   Computes an inner product between the r-ad expectation densities given by
%   the weighted pitch multisets (x_p, x_w) and (y_p, y_w), and with normal
%   perception error with standard deviation sigma. Four different inner
%   products can be computed, depending on the penultimate two arguments: the
%   inner product assumes periodic equivalence with period set by 'period' if
%   isPer == 1, and assumes transpositional equivalence (relative rather than
%   absolute pitches) if isRel == true.
%
%   See the expectationTensor function for further information about these
%   parameters.
%
%   Originally by David Bulger, Macquarie University, Australia (2016). (Andrew
%   J. Milne, The MARCS Institute, Western Sydney University made a few trivial
%   changes to comments and variable names for consistency with other routines
%   in the Music Perception Toolbox.)

if r > min(numel(x_p),numel(y_p))
    s = NaN;
else
    J = period;
%     if isempty(x_w) || isequal(x_w,0)
%         x_w = ones(numel(x_p),1);
%     end
%     if isempty(y_w) || isequal(y_w,0)
%         y_w = ones(numel(y_p),1);
%     end
    
    x_p = x_p(:);
    x_w = x_w(:);
    y_p = y_p(:);
    y_w = y_w(:);
        
    if isempty(x_w)
        x_w = ones(numel(x_p),1);
    end
    if numel(x_w) == 1
        if x_w == 0
            warning('All weights in x_w are zero.');
        end            
        x_w = x_w*ones(numel(x_p),1);
    end
    
    if isempty(y_w)
        y_w = ones(numel(y_p),1);
    end
    if numel(y_w) == 1
        if y_w == 0
            warning('All weights in x_w are zero.');
        end            
        y_w = y_w*ones(numel(y_p),1);
    end
    
    if rem(r,1) || r<1
        error('''r'' must be a positive integer.');
    elseif r > min(numel(x_p),numel(y_p))
        error(['''r'' must not exceed the cardinality of either pitch ' ...
            'multiset.']);
    elseif numel(x_p)~=numel(x_w) || numel(y_p)~=numel(y_w)
        error(['x_w must have the same number of entries as x_p; ' ...
            'likewise for y_w and y_p.']);
    end
    
    if isRel
        if r < 2
            error('For relative densities, ''r'' must be at least 2.');
        end
        RM = eye(r) - ones(r)/r;  % Riemannian metric
    else
        % Absolute densities.
        RM = eye(r);
    end
    
    s = ip(x_p,x_w,y_p,y_w) / sqrt(ip(x_p,x_w,x_p,x_w)*ip(y_p,y_w,y_p,y_w));
end
    function ipval = ip(u_p,u_w,v_p,v_w)
        count = 0;
        % Note, this is a "nested" function, so RM, r, and sigma are still in
        % scope.
        ipval = 0;
        % Construct all permutations without replacement of size r from
        % 1:numel(u_p). (Due to symmetry, we don't need to do this for both
        % densities; just looking at the ordered combinations (from nchoosek)
        % suffices for one density (the one from v).)
        nck = nchoosek(1:numel(u_p), r)';
        npermk = [];
        for shuff = perms(1:r)'
            npermk = [npermk, nck(shuff,:)];
        end
        for j = npermk
            for k = nchoosek(1:numel(v_p), r)'
                diff = u_p(j) - v_p(k);  % "difference vector"
                if isPer
                    diff = mod(diff + J/2,J) - J/2;
                end
                ipval = ipval ...
                    + prod([u_w(j); v_w(k)]) ...
                    * exp(-(diff'*RM*diff)/(4*sigma^2)); % This more general
                % approach could be a performance hit in the absolute case;
                % keep an eye on it.
                count = count+1;
            end
        end
    end
end

