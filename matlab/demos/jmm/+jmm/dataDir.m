function d = dataDir()
%DATADIR  The demos/jmm/data folder (the bundled BWV 347 MusicXML lives here).
%
%   d = jmm.dataDir()
    d = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data');
end
