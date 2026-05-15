function tf = errorMessageContains(fn, expectedSubstr)
%ERRORMESSAGECONTAINS  Test helper: returns true if fn() throws an error whose message contains a substring.
%
%   tf = errorMessageContains(@() someFunction(badArgs), 'expected text')
%
%   Returns true if the supplied function handle throws an error
%   whose message contains expectedSubstr, false otherwise
%   (including the case of no error). Used in MPT's test suite when
%   the error identifier is unstable or unset but a substring of
%   the message is reliable.
%
%   See also: throwsError, throwsErrorWithId.

    try
        fn();
        tf = false;
    catch ME
        tf = ~isempty(strfind(ME.message, expectedSubstr)); %#ok<STREMP>
    end
end
