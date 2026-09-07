function d = dataDir()
%DATADIR  The demos/jmm/data folder (the bundled BWV 347 MusicXML lives here).
%
%   d = jmm.dataDir()
%
%   Twin of jmm_data.DATA_DIR in the Python demos.
    d = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data');
end
