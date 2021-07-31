function [X,pVals] = expectationTensor(x_p, x_w, sigma, kerLen, ...
                                       r, isRel, isPer, limits, ...
                                       isSparse, doPlot, tol)
%EXPECTATIONTENSOR Expectation tensor for r-ads in a weighted pitch multiset.
%
%   [X, pVals] = expectationTensor(x_p, x_w, sigma, kerLen, r, isRel, isPer,
%   limits, isSparse, doPlot): X is the hypercubic expectation tensor for r-ads
%   in a weighted pitch multiset; pVals are the pitch values for all dimensions
%   of the tensor. The tensor can be smoothed or unsmoothed, periodic or
%   nonperiodic, absolute or relative.
%
%   x_p = the pitch multiset.
%
%   x_w = saliences/weights of the pitches in x_p.
%
%   sigma = standard deviation of the discrete Gaussian in the units of x_p.
%
%   kerLen = the length of the smoothing kernel in standard deviations. The
%   kernel length is adjusted so it contains an odd number of entries so its
%   mode is the central value.
%
%   r = 1, 2, 3, or 4 - i.e., monad, dyad, triad, or tetrad representation.
%
%   isRel = transpositional invariance (1) or not (0).
%
%   isPer = 0 or 1 - make the tensor not periodic or periodic; that is, treat
%   pitches as pitch classes, modulo the period, when periodic is 1.
%
%   limits = a scalar or a 2-entry row vector: if isPer == 0, limits sets the
%   interval of pitches over which the expectation tensor is generated
%   (expanded to inlcude the tails of the smoothing kernel applied to those
%   within-limits pitches); if isPer == 1, the last entry of limits gives the
%   period of repetition (e.g., if the units of x_p are cents, limits = 1200
%   gives periodicity at the octave). The pitch values of the rows, columns,
%   pages, etc. in X are given in pVals.
%
%   isSparse = 0 or 1 - when r = 1 or r = 2 and isRel = 1, the resulting
%   expectation vector (or scalar) is always full. All other cases return a
%   sparse array structure when isSparse = 1 (the default); otherwise, it is
%   converted to a full array. For high dimensional tensors, the conversion to
%   full format may be slow and may exceed available memory.
%
%   doPlot = 0 or 1 - make or do not make a plot of the resulting tensor. When
%   the tensor has dimensionality greater than 2, the higher dimensions are
%   summed over. Setting doPlot to 1 forces isSparse to be 0.
%
%   tol = all values below this are made zero (removed from sparse array) of 
%   the unsmoothed expectation tensor prior to convolution. This can
%   considerably speed up the calculation of the smoothed tensor. The default
%   is 0.00001. It may be useful to set tol = 0 if the smoothed tensor is later
%   log transformed.
%
%   Note that the procedures run at O(I(J^r)), where I is the number of
%   elements in x_p, and J is the number of pitch units spanned by x_p. For
%   higher values of r or I, it may be necessary to reduce J. High values of r
%   and J may also lead to "out of memory" errors (even when sparse array
%   structures are used), due to the huge size of the resulting arrays.
%
%   References: Milne, A.J., Sethares, W.A., Laney, R., Sharp, D.B. (2011)
%   Modelling the Similarity of Pitch Collections with Expectation Tensors,
%   Journal of Mathematics and Music, 5(1), 1-20.
%
%   Milne, A. J. (2013). A Computational Model of the Cognition of Tonality.
%   PhD thesis, The Open University. Chapter 3.
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University.

persistent sigmaLast kerLenLast rLast isRelLast isPerLast limitsLast ...
    FgKer x2_e_k_2 gKer gKerLen spKer gKerDotProd

if nargin < 11
    tol = 0.00001;
end
if nargin < 10
    doPlot = 0;
end
if nargin < 9
    isSparse = 1;
end
if nargin < 8
    limits = [0 1200];
end
if nargin < 7
    isPer = 1;
end
if nargin < 6
    r = 1;
end
if nargin < 5
    isRel = 0;
end

if rem(r,1) || r<1
    error('r must be a positive integer.')
end
if sigma < 0
    error('sigma must be nonnegative.')
end
if doPlot==1 && isSparse==1
    warning(['isSparse has been changed to 0 in order to allow plots to ' ...
             'be drawn; doPlot must be 0 if you want sparse output.'])
end

if ~isequal(sigma,sigmaLast) || ~isequal(kerLen,kerLenLast) ...
        || ~isequal(isRel,isRelLast) || ~isequal(r,rLast) ...
        || ~isequal(isPer,isPerLast)
    newKer = true;
else
    newKer = false;
end
if ~isequal(limits,limitsLast)
    newLim = true;
else
    newLim = false;
end

%% Fixed parameters
normalize = 1; % make modes in expectation tensors equal to counts.

%% Preliminaries
nDimX = r-isRel;

%% Generate smoothing kernel
% Create nDimX-dimensional Gaussian kernel (sparse if nDimX > 1)
if newKer
    if sigma < 0
        error('Sigma must be non-negative.')
    end
    if sigma == 0
        sigma = eps;
        warning(['Sigma must be greater than zero; it has been set to ' ... 
                 '2.2204e-16.'])
    end
    if nDimX > 1 || nDimX==1 && isPer==0
        SIG = sigma^2 * eye(nDimX) * 2^isRel; % variance of difference
        % distributions (when isRel==1) is scaled by 2
    end
    if nDimX == 0 
        SIG = 0;
    end
    
    K = ceil(sigma*kerLen);
    if bitget(K,1) == 1 % 1 if K is odd
        K = K - 1;
    end
    k = 0:K;
    gKerLen = numel(k);
    
    if nDimX == 1
        if isRel==1 && isPer==0
            gKer = normpdf(k',ceil(K/2), sigma * sqrt(2^isRel))'; % standard 
            % deviation of a difference distribution (i.e., when isRel==1) is 
            % scaled by sqrt(2)
            spKer = array2SpArray(gKer);
        else
            gKer = sqrt(2 * pi * (sigma * sqrt(2^isRel))^2) ...
                 * normpdf(k',ceil(K/2), sigma)'; % standard deviation of a
            % difference distribution (i.e., when isRel==1) is scaled by
            % sqrt(2)
            if normalize == 1
                gKerDotProd = gKer*gKer';
            else
                gKerDotProd = 1;
            end
        end
    elseif nDimX == 2
        [X1,X2] = ndgrid(k,k);
        ker = mvnpdf([X1(:) X2(:)],ceil(K/2),SIG);
        ker = reshape(ker,[gKerLen gKerLen]);
        spKer = array2SpArray(ker);
    elseif nDimX == 3
        [X1,X2,X3] = ndgrid(k,k,k);
        ker = mvnpdf([X1(:) X2(:) X3(:)],ceil(K/2),SIG);
        ker = reshape(ker,[gKerLen gKerLen gKerLen]);
        spKer = array2SpArray(ker);
    elseif nDimX == 4
        [X1,X2,X3,X4] = ndgrid(k,k,k,k);
        ker = mvnpdf([X1(:) X2(:) X3(:) X4(:)],ceil(K/2),SIG);
        ker = reshape(ker,[gKerLen gKerLen gKerLen gKerLen]);
        spKer = array2SpArray(ker);
    end
    if (nDimX>1 || (isRel==1 && isPer==0)) && normalize==1 % CHECK THIS!!
%    if nDimX>1 && normalize==1
        spKer = spTimes(sqrt(det(2*pi*SIG)),spKer);
    end
end
% Note that gKerLen is always odd: offset gives the number of kernel entries,
% in any single dimension, before or after the kernel's central value
halfKerLen = (gKerLen-1)/2;

%% Transforms of x_p, x_w, y_p, and y_w 
% Remove invalid x_p and x_w values
x_p = x_p(:);
finInd = isfinite(x_p);
x_p = x_p(finInd);
x_p = round(x_p);
I = numel(x_p);

if numel(x_w) > 1
    x_w = x_w(finInd);
    % Error check
    if I ~= numel(x_w)
        error('x_p and x_w, must have the same number of (finite) entries.')
    end
elseif numel(x_w) == 1
    if x_w == 0
        warning('All weights in x_w are zero.');
    end
    x_w = x_w*ones(I,1);
elseif isempty(x_w)
    x_w = ones(I,1);
end
x_w = x_w(:);

limits = round(limits);
if numel(limits) == 1
    limits(2) = limits(1);
    limits(1) = 0;
end
if isempty(limits) 
    if isRel == 0
        limits = [min(x_p) max(x_p)];
    else
        error('A "limits" argument must be entered for periodic tensors.')
    end
end

% Change x_p and x_w in light of isRel, isPer, and limits arguments: If
% nonperiodic and absolute remove all pitches outside limits (taking into 
% account the kernel width)
if isRel==0 && isPer==0
    if numel(x_p(x_p<limits(1)-gKerLen | x_p>limits(2)+gKerLen)) == numel(x_p)
        error(['All pitches have been removed from x_p because they ' ...
               'all lie outside the range set by "limits".'])
    end
    if numel(x_p(x_p<limits(1)-gKerLen | x_p>limits(2)+gKerLen)) > 0
        warning(['Some pitches in x_p lie outside the range set by ' ... 
                 '"limits", hence they have been removed.'])
    end
    x_w(x_p<limits(1)-gKerLen | x_p>limits(2)+gKerLen) = [];
    x_p(x_p<limits(1)-gKerLen | x_p>limits(2)+gKerLen) = [];
    I = numel(x_p);
end

if isPer == 1
    J = limits(2);
    if J < 0
        error('For periodic tensors, the last entry of "limits" must be greater than 0.')
    end
    % Convert x_p to x_p modulo the period
    x_p = mod(x_p,J);
    offset = 0;
else % isPer = 0
    J = limits(2)-limits(1);
    if size(limits,1) ~= 1 || size(limits,2) ~= 2
        error('For nonperiodic tensors, "limits" must be a 2-entry row vector.')
    end
    % Offset to make lowest x_p equal 0
    offset = min(x_p);
    x_p = x_p - offset;
end

%% Get pVals in light of isPer, limits, and kernel
if isPer == 0
    pLo = limits(1) - halfKerLen;
    pHi = limits(2) + halfKerLen;
else
    pLo = 0;
    pHi = limits(end) - 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pVals = (pLo:pHi)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Return all-zeros X if r > I
dimX = repelem(J,nDimX); % Get size of X
if r > I
    if isSparse == 0
        if nDimX > 1
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            X = zeros(dimX);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            X = zeros([dimX 1]);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    else
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        X = struct('Size',dimX,'Ind',[],'Val',[]);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    return
end

%% Pitch matrices X_p_ij (smoothed) or X_ij (unsmoothed) and their column sum
if (r==1 && isRel==0) || (r==2 && isRel==1 && isPer==1)
    lowInd = x_p + 1;
    highInd = x_p + gKerLen;
    gKer_w = x_w*gKer;
    X_p_ij = zeros(I,J+gKerLen);
    for i = 1:I % Using a loop is faster than indexing that avoids looping
        X_p_ij(i,lowInd(i):highInd(i)) = gKer_w(i,:);
    end
    % Shift/wrap 
    X_p_ij(:,1:gKerLen) ...
        = X_p_ij(:,1:gKerLen) + isPer*X_p_ij(:,J+1 : J+gKerLen);
    if r == 1
        % Sum
        X_p_j = sum(X_p_ij);
        if isPer == 1
            % Remove excess
            X_p_j = X_p_j(:,1:J);
        end
    elseif r == 2
        % Remove excess
        if isPer == 1
            X_p_ij = X_p_ij(:,1:J);
        end
        % Sum
        X_p_j = sum(X_p_ij);
    end
else
    X_ij = zeros(I,J);
    for i = 1:I % Using a loop is faster than indexing that avoids looping
        X_ij(i,x_p(i)+1) = x_w(i);
    end
    X_j = sum(X_ij);
end


if r == 1
    %% Abs/RelMonadExp
    % Represent a pitch (class) multiset as an absolute expectation vector or 
    % as a relative expectation scalar.

    if isRel == 0
        % Smoothed absolute monad vector
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        X = X_p_j';
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Shifted to line up with pVals
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        X = circshift(X,offset-halfKerLen-pLo); 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else % isRel == 1
        % Relative monad scalar
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        X = sum(x_w);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    
elseif r == 2
    %% Abs/RelDyadExp
    % Represent a pitch/time (class) multiset as an absolute dyad expectation
    % tensor or a relative dyad epectation vector.
    
    if isRel==1 && isPer==1
        % Note that this routine uses circular convolution (calculated with
        % FFTs) because this is faster than the methods used below (this method
        % is only suitable for this set of features).

        % Circular autocorrelation
        x1_e_k_2 = ifft(abs(fft(X_p_j', J)).^2, J); % abs(x).^2 is faster than
        % x.*conj(x)
        x1_e_k_2 = x1_e_k_2/gKerDotProd;
        x1_e_k_2(x1_e_k_2<1e-15) = 0;
        if newKer || newLim
            FgKer = fft(gKer', J);
            FgKer = FgKer(:); % this is required because fft of a scalar gKer
            % (e.g., when sigma or kerLen are 0) returns a row instead of 
            % column vector
            x2_e_k_2 = ifft(abs(FgKer).^2, J);
            x2_e_k_2 = x2_e_k_2/gKerDotProd;
        end
        % Smoothed periodic relative dyad vector
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        X = x1_e_k_2 - (x_w' * x_w) * x2_e_k_2;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else % isRel==0 || isPer==0
        % Term 1: Make sparse and do the outer product
        spX_j = array2SpArray(X_j);
        term1 = spOuter(spX_j, spX_j);
        % Term 2
        outX_ij = cell(1,I);
        for i = 1:I
            spX_iji = array2SpArray(X_ij(i,:));
            outX_ij{i} = spOuter(spX_iji, spX_iji);
        end
        % Sum them to make the generalized Khatri-Rao product
        sumOutX_ij = spPlus(outX_ij);
        
        term2 = spTimes(-1,sumOutX_ij); % multiply by -1
        
        % If isPer==1 & isRel==0, unsmoothed periodic absolute dyad expectation
        % matrix (sparse)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        X = spPlus(term1,term2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end


elseif r == 3
    %% Abs/RelTriadExp
    % Represent a pitch/time (class) multiset as an absolute triad expectation
    % tensor or a relative triad epectation matrix.
    
    % Term 1: outer product of X_j
    % Make sparse and do the outer product
    spX_j = array2SpArray(X_j);
    term1 = spOuter(spX_j,spX_j,spX_j);
    
    % Term2: Generalized Khatri-Rao products and outer product
    % Outer products (squares) of rows of X_ij
    outSpX_ij = cell(1,I);
    for i = 1:I
        spX_ij = array2SpArray(X_ij(i,:));
        outSpX_ij{i} = spOuter(spX_ij, spX_ij);
    end
    % Sum them to make the generalized Khatri-Rao product
    sumOutSpX_ij = spPlus(outSpX_ij);
    
    % Outer product with results of KR product and relevant permutations
    term2_123 = spOuter(spX_j,sumOutSpX_ij);
    term2_213 = spPerm(term2_123,[2 1 3]);
    term2_231 = spPerm(term2_123,[2 3 1]);
    
    % Sum the permutations and multiply by -1
    term2 = spTimes(-1,spPlus(term2_123,term2_213,term2_231));
    
    % Term 3: Generalized Khatri-Rao products
    % Outer products (cubes) of rows of X_ij
    for i = 1:I
        spX_ij = array2SpArray(X_ij(i,:));
        outSpX_ij{i} = spOuter(spX_ij, spX_ij, spX_ij);
    end
    % Sum them to make the generalized Khatri-Rao product
    sumOutSpX_ij = spPlus(outSpX_ij);
    
    term3 = spTimes(2,sumOutSpX_ij);
    
    % If isPer==1, unsmoothed periodic absolute triad expectation tensor 
    % (sparse)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    X = spPlus(term1,term2,term3);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
elseif r == 4
    %% AbsTetradExp
    % Represent a pitch/time (class) multiset as an absolute or relative tetrad
    % expectation tensor.
    
    % Term 1: Outer product of X_j
    % make sparse and do the outer product
    spX_j = array2SpArray(X_j);
    term1 = spOuter(spX_j, spX_j, spX_j, spX_j);
    
    % Term2: Generalized Khatri-Rao products and outer product
    % Outer products (squares) of rows of X_ij
    outSpX_ij = cell(1,I);
    for i = 1:I
        spX_ij = array2SpArray(X_ij(i,:));
        outSpX_ij{i} = spOuter(spX_ij, spX_ij);
    end
    % Sum them to make the generalized Khatri-Rao product
    sumOutSpX_ij = spPlus(outSpX_ij);
    
    % Outer product of X_j and KR product and relevant permutations
    term2_1234 = spOuter(spX_j, spX_j, sumOutSpX_ij);
    term2_1324 = spPerm(term2_1234,[1 3 2 4]);
    term2_1342 = spPerm(term2_1234,[1 3 4 2]);
    term2_3124 = spPerm(term2_1234,[3 1 2 4]);
    term2_3142 = spPerm(term2_1234,[3 1 4 2]);
    term2_3412 = spPerm(term2_1234,[3 4 1 2]);
    
    % Sum the permutations and multiply by -1
    term2 = spTimes(-1,spPlus(term2_1234,term2_1324,term2_1342,...
        term2_3124,term2_3142,term2_3412));
    
    % Term3: Generalized Khatri-Rao products and outer product
    % Outer products (cubes) of rows of X_ij
    outSpX_ij = cell(1,I);
    for i = 1:I
        spX_ij = array2SpArray(X_ij(i,:));
        outSpX_ij{i} = spOuter(spX_ij, spX_ij, spX_ij);
    end
    % Sum them to make the generalized Khatri-Rao product
    sumOutSpX_ij = spPlus(outSpX_ij);
    
    % Outer product of X_j and KR product and relevant permutations
    term3_1234 = spOuter(spX_j, sumOutSpX_ij);
    term3_2134 = spPerm(term3_1234,[2 1 3 4]);
    term3_2314 = spPerm(term3_1234,[2 3 1 4]);
    term3_2341 = spPerm(term3_1234,[2 3 4 1]);
    
    % Sum the permutations and multiply by 2
    term3 = spTimes(2,spPlus(term3_1234,term3_2134,term3_2314,term3_2341));
    
    % Term4: Generalized Khatri-Rao products and outer product
    % Outer products (squares) of rows of X_ij
    outSpX_ij = cell(1,I);
    for i = 1:I
        spX_ij = array2SpArray(X_ij(i,:));
        outSpX_ij{i} = spOuter(spX_ij, spX_ij);
    end
    % Sum them to make the generalized Khatri-Rao product
    sumOutSpX_ij = spPlus(outSpX_ij);
    
    % Outer product (square) of KR product and relevant permutations
    term4_1234 = spOuter(sumOutSpX_ij, sumOutSpX_ij);
    term4_1324 = spPerm(term4_1234,[1 3 2 4]);
    term4_1342 = spPerm(term4_1234,[1 3 4 2]);
    
    % Sum the permutations and multiply by 2
    term4 = spPlus(term4_1234,term4_1324,term4_1342);
    
    % Term 5: Outer product for each row of X_ij
    outSpX_ij = cell(1,I);
    for i = 1:I
        X_iji = array2SpArray(X_ij(i,:));
        outSpX_ij{i} = spOuter(X_iji, X_iji, X_iji, X_iji);
    end
    term5 = spTimes(-6,spPlus(outSpX_ij)); % sum and multiply by -6
    
    % If isPer==1, unsmoothed periodic absolute tetrad expectation tensor 
    % (sparse)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    X = spPlus(term1, term2, term3, term4, term5);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        

else
    error('Function does not currently exist - use a lower value for "r".');
end

%% Build the expectation tensors
if nDimX > 1 || (nDimX==1 && isPer==0 && isRel==1)
    if gKerLen > 1
        X = spTol(X,tol);
        spKer = spTol(spKer,tol);
    end
    
    if isRel == 0
        if isPer == 0
            % Truncate/pad to match limits
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            X = spTrunc(repelem(pLo-halfKerLen-offset+1,nDimX), ...
                        repelem(pHi+halfKerLen-offset+1,nDimX),X);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if gKerLen > 1
                X = spConv(X,spKer,'full');            
                % Negative noncircular shift achieved through trunction
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                X = spTrunc(repelem(gKerLen,nDimX), ...
                            repelem(gKerLen+pHi-pLo,nDimX),X);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
        else % isPer == 1
            if gKerLen > 1
                X = spConv(X,spKer,'circ');
                % Negative circular shift to line up with pVals after 
                % convolution
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                X = spShift(X,repelem(-halfKerLen,nDimX),isPer);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
        end
    else % isRel == 1
        shifts = [repelem(-1,nDimX) 0];
        isProg = 1;
        collapse = 1;
        % Progressive shift and sum to make absolute tensor relative
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        X = spShift(X,shifts,isPer,isProg,collapse); 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if isPer == 0
            if gKerLen == 1
                % Truncate/pad to match limits
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                X = spTrunc(repelem(pLo+max(x_p)+1,nDimX), ...
                            repelem(pHi+max(x_p)+1,nDimX),X); 
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            else % gKerLen > 1
                X = spConv(X,spKer,'full');            
                % Truncate/remove all entries outside pLo - offset and pHi +
                % gKerLen (entries pLo and and pLo - offset, and entries
                % between pHi and pHi + offset are removed after convolution)
                X = spTrunc(repelem(pLo-halfKerLen+max(x_p)+1,nDimX), ...
                            repelem(pHi+halfKerLen+max(x_p)+1,nDimX),X); 
                % Negative noncircular shift achieved through truncation and
                % also truncating outside limits
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                X = spTrunc(repelem(gKerLen,nDimX), ...
                            repelem(pHi-pLo+gKerLen,nDimX),X);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
        else % isPer == 1
            if gKerLen > 1
                X = spConv(X,spKer,'circ');
                % Negative circ shift to line up with pVals after convolution
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                X = spShift(X,repelem(-halfKerLen,nDimX),isPer); 
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
            end
        end
    end
        
    if isempty(X.Ind)
        warning('There are no nonzero expectations within the "limits" specified in the argument.')
        if isSparse == 1
            return
        else
            % Empty expectation tensor (sparse)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            X = zeros(dimX);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            return
        end
    end
        
    if isSparse==0 
        % Tensor (full)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        X = spArray2Array(X);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

%% Plots
if doPlot == 1
    if isSparse==1
        plotX = spArray2Array(X);
    else
        plotX = X;
    end
    
    figNum = (r-1)*4 + isRel*2 + isPer + 1;
    
    if r == 1
        figNameR = 'monad';
    elseif r == 2
        figNameR = 'dyad';
    elseif r == 3
        figNameR = 'triad';
    elseif r == 4
        figNameR = 'tetrad';
    end
    
    if isRel == 0
        figNameA = 'absolute ';
    else
        figNameA = 'relative ';
    end
    
    if isPer == 0
        figNameP = 'Nonperiodic ';
    else
        figNameP = 'Periodic ';
    end
    
    if nDimX == 0
        figNameT = ' expectation scalar';
    elseif nDimX == 1
        figNameT = ' expectation vector';
    elseif nDimX == 2
        figNameT = ' expectation matrix';
    else
        figNameT = ' expectation tensor';
    end
    
    % Tick gaps for the plots
    if J < 30
        tickGap = 1;
    elseif J < 300
        tickGap = 10;
    elseif J < 3000
        tickGap = 100;
    elseif J < 12000
        tickGap = 400;
    elseif J < 36000
        tickGap = 1200;
    end
    
    tickVals ...
        = [ceil(pVals(1)/tickGap)*tickGap ceil(pVals(end)/tickGap)*tickGap];    
    switch nDimX
        case 1
            if isPer==1 
                plotX = [plotX; plotX(1)];
            else
                plotX = [plotX; 0];
            end
        case 2
            if isPer==1
                plotX = [plotX plotX(:,1); plotX(1,:) 0];
            else
                plotX = [plotX 0*plotX(:,1); 0*plotX(1,:) 0]; 
            end
        case 3
            plotX = squeeze(sum(plotX,3));
            if isPer==1
                plotX = [plotX plotX(:,1); plotX(1,:) 0];
            else
                plotX = [plotX 0*plotX(:,1); 0*plotX(1,:) 0]; 
            end
        case 4
            plotX = squeeze(sum(sum(plotX,4),3));
            if isPer==1
                plotX = [plotX plotX(:,1); plotX(1,:) 0];
            else
                plotX = [plotX 0*plotX(:,1); 0*plotX(1,:) 0]; 
            end
    end
    
    plotX = plotX/sum(plotX(1 : end-1)); %%%%% !!!!!! %%%%%%
    
    if figNum ~= 3 && figNum ~= 4
        if nDimX == 1
            % hold on
            figure(figNum)
            stairs([pVals; pVals(end)+1],plotX,'LineWidth',2,'LineStyle','-')
            axis([pVals(1) pVals(end)+1 0,max(plotX)*1.1])
            set(gca,'XTick',tickVals(1):tickGap:tickVals(2))
            ax = gca;
            ax.FontSize = 16;
            ax.XLabel.String = 'Log frequency (cents)';
        elseif nDimX > 1
            % hold off
            figure(figNum)
            surf([pVals; pVals(end)+1],[pVals; pVals(end)+1], plotX, ...
                'FaceAlpha',1,'LineStyle','none')
            axis([pVals(1) pVals(end)+1 pVals(1) pVals(end)+1])
            set(gca,'XTick',tickVals(1):tickGap:tickVals(2))
            set(gca,'YTick',tickVals(1):tickGap:tickVals(2))
            set(gca,'color',[0.8 0.8 0.8])
            ax = gca;
            ax.FontSize = 16;
            ax.CLim = 1200*[0,max(plotX(:)/50)]/J;
            colormap bone
            lighting phong
            grid off
            axis square
        end
        title([figNameP figNameA figNameR figNameT], 'Fontweight','normal')
    end
    clear plotX
end

%% Store last-used values
sigmaLast = sigma;
kerLenLast = kerLen;
rLast = r;
isRelLast = isRel;
isPerLast = isPer;
limitsLast = limits;

end
