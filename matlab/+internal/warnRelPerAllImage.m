function warnRelPerAllImage(varargin) %#ok<INUSD>
%WARNRELPERALLIMAGE  Retired.
%
%   Retained as a no-op for backward compatibility with any external
%   call sites. The substitution this once warned about --- rel-per
%   dispatch giving the "faster all-image form" instead of the
%   "canonical single-wrap measure" --- no longer describes reality:
%   rel-per full-image (C) is the toolbox's default measure and the
%   ``wrap = 'single-image'`` opt-in gives (A) explicitly.
%
%   Mirror of Python dispatch._warn_rel_per_all_image (also retired).
end
