function tf = throwsErrorWithId(fn, expectedId)
%THROWSERRORWITHID  Test helper: returns true if fn() throws an error with a matching identifier.
%
%   tf = throwsErrorWithId(@() someFunction(badArgs), 'package:badInput')
%
%   Returns true if the supplied function handle throws an error
%   whose identifier matches expectedId exactly, false otherwise
%   (including the case of no error). Used in MPT's test suite to
%   pin down specific error identifiers rather than merely
%   "something raised".
%
%   See also: throwsError, errorMessageContains.

    try
        fn();
        tf = false;
    catch ME
        tf = strcmp(ME.identifier, expectedId);
    end
end
