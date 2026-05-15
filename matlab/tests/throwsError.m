function tf = throwsError(fn)
%THROWSERROR  Test helper: returns true if fn() throws any error.
%
%   tf = throwsError(@() someFunction(badArgs))
%
%   Returns true if the supplied function handle throws an error
%   when invoked, false if it returns normally. Used pervasively in
%   MPT's test suite for negative-path assertions.
%
%   See also: throwsErrorWithId, errorMessageContains.

    try
        fn();
        tf = false;
    catch
        tf = true;
    end
end
