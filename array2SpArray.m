function spA = array2SpArray(A)
%ARRAY2SPARRAY Convert a full array into a sparse array structure.
%
%   spA = array2spArray(A) A sparse array structure has the following fields:
%
%   'Size', which is a row vector of the sizes of each dimension of the full
%   array -- for example, a column vector with N entries has a size of N; a row
%   vector with N entries has a size of (1,N); a matrix with N entries has a
%   size (J,K), where JK = N; a three-way array with N entries has a size
%   (J,K,L), where JKL = N;
%
%   'Ind', which is a column vector of linear indices of nonzero values in the
%   full array that they represent;
%
%   'Val', which is a column vector of the values at those indices.
%
%   All singleton dimensions are removed.
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University
%
%   See also SPARRAY2ARRAY.

[indA,~,valA] = find(A(:)); % find nonzeros in full array
sizA = size(A); % get size of full array
sizA(sizA==1) = []; % remove all singleton dimensions
if isempty(sizA)
    sizA = [];
end

% Create structure
spA = struct('Size',sizA,'Ind',indA,'Val',valA);

end
