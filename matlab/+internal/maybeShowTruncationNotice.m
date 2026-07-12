function maybeShowTruncationNotice(cmd)
%INTERNAL.MAYBESHOWTRUNCATIONNOTICE  One-time warning: truncation is on by default.
%
%   internal.maybeShowTruncationNotice() issues, at most once per MATLAB
%   session, a warning that kernel evaluation truncates at 6 sigma
%   by default. It is called from the mptDefaults('truncationSigmas')
%   getter, through which every kernel-evaluating path resolves the
%   default, so it fires on the first kernel evaluation regardless of
%   which public function the user's script calls. The notice prints
%   when both of:
%     - it has not already fired this session, and
%     - the toolbox-wide truncationSigmas default is at its factory
%       value of 6 (so it stays silent once the user sets their own
%       truncation, and throughout the test suite, which pins Inf).
%
%   It is issued as a warning with identifier 'mpt:truncationDefault'
%   (goes to stderr, suppressible via
%   warning('off', 'mpt:truncationDefault')) rather than printed to
%   stdout, so it does not land in the middle of a script's own printed
%   output. It is deliberately NOT gated by showHints: the notice always
%   gets its single showing, so a script that sets showHints = false to
%   quiet the dispatch-decision messages still sees the notice once.
%   showHints governs only the dispatch messages.
%
%   internal.maybeShowTruncationNotice('suppress') marks the notice as
%   already shown without printing; internal.maybeShowTruncationNotice
%   ('rearm') clears that flag. The test-suite isolation helper
%   (mptTestIsolateDefaults) suppresses on entry and re-arms on
%   teardown, so running the suite does not permanently silence the
%   notice for the rest of the session.
%
%   The once-per-session flag is a persistent, cleared by `clear all`
%   or a fresh MATLAB session, so the notice reappears on reopen. It is
%   NOT re-armed by mptDefaults('reset').
%
%   See also: MPTDEFAULTS.

    persistent shown
    if isempty(shown)
        shown = false;
    end

    if nargin >= 1 && (ischar(cmd) || isstring(cmd))
        switch lower(char(cmd))
            case 'suppress'
                shown = true;
                return;
            case 'rearm'
                shown = false;
                return;
        end
    end

    if shown
        return;
    end

    S = mptDefaults();
    if ~(isfield(S, 'truncationSigmas') && isnumeric(S.truncationSigmas) ...
            && isscalar(S.truncationSigmas) && S.truncationSigmas == 6)
        return;
    end

    shown = true;
    warning('mpt:truncationDefault', ...
        ['Kernel evaluation truncates the Gaussian kernel at 6 sigma ' ...
         'by default, which runs much faster than the exact sum; the ' ...
         'speed-up grows with tuple size r and multiset size, where ' ...
         'the exact sum has many kernel centres and becomes expensive. ' ...
         'Worst-case error vs the exact result is about 2e-8 at 6 ' ...
         'sigma (the default), ~1e-5 at 5, and ~1e-3 at 4. Set ' ...
         'mptDefaults(''truncationSigmas'', Inf) for the exact result. ' ...
         'This warning shows only once per session.']);
end
