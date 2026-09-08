function out = writePreMaet(varargin)
%WRITEPREMAET  Write a pre-MAET as CSV, in the format readPreMaet reads.
%
%   WRITEPREMAET(DESTINATION, PM) and
%   WRITEPREMAET(DESTINATION, PATTR, WATTR, SPECS) write the pre-MAET as a
%   table a spreadsheet can open: one row per attribute, its parameters at
%   the left and one column per event (Milne 2026, Def. 2.6). DESTINATION
%   is a path, or [] to return the CSV text without writing it.
%
%   The rendering is showPreMaet's, under 'format', 'csv', so the cells
%   carry the notation of the article and the file is what readPreMaet
%   reads: a pre-MAET survives a round trip through a spreadsheet
%   unchanged. Every showPreMaet argument is accepted; a file carries the
%   whole pre-MAET, so no events or elements are elided.
%
%   STR = WRITEPREMAET(...) also returns the CSV text.
%
%   See also READPREMAET, SHOWPREMAET, BUILDEXPTENS.

varargin = internal.expandPreMaet(varargin, 2);
out = localWritePreMaet(varargin{:});
end


function out = localWritePreMaet(destination, pAttr, w, specs, varargin)
if nargin < 3, w = []; end
if nargin < 4, specs = []; end
text = showPreMaet(pAttr, w, specs, varargin{:}, ...
                   'format', 'csv', 'verbose', false);
if ~isempty(destination)
    fid = fopen(destination, 'w');
    if fid < 0
        error('writePreMaet:open', 'Cannot open %s for writing.', ...
              char(destination));
    end
    fwrite(fid, text);
    fclose(fid);
end
if nargout > 0 || isempty(destination)
    out = text;
end
end
