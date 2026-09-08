%% test_pre_maet_io.m — kernel geometry in specs, and CSV I/O
%
%  Two related things are tested here. First, that a spec may carry the
%  attribute's sigma, isPer and period -- the parameters Milne (2026,
%  Def. 2.6) counts as part of the pre-MAET -- that an explicit keyword
%  overrides them, and that a value missing from both places is refused
%  by name. Second, that a pre-MAET survives a round trip through CSV
%  unchanged, the writer and the reader sharing one cell grammar.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

pIO = {[60 62 64], [0 1 2]};
mkSpecs = @() {struct('name','pitch','r',1,'rel',false,'sym',true, ...
                      'sigma',0.5,'isPer',true,'period',12), ...
               struct('name','time','r',1,'rel',false,'sym',true, ...
                      'sigma',0.25,'isPer',false,'period',0)};

% ---- Resolution ----

d = buildExpTens(pIO, [], 'specs', mkSpecs(), 'verbose', false);
results{end+1,1} = 'preMaetIo: specs alone suffice'; %#ok<SAGROW>
results{end,2}   = abs(d.sigma(1) - 0.5) < 1e-12 && ...
                   abs(d.sigma(2) - 0.25) < 1e-12 && ...
                   d.isPer(1) && ~d.isPer(2);

% A sweep supplies sigma per call while the specs hold a baseline, so the
% two disagreeing is the ordinary idiom, not an error.
d = buildExpTens(pIO, [], 'specs', mkSpecs(), 'sigma', [9 9], 'verbose', false);
results{end+1,1} = 'preMaetIo: keyword overrides silently'; %#ok<SAGROW>
results{end,2}   = all(abs(d.sigma - 9) < 1e-12);

sp = mkSpecs(); sp{2} = rmfield(sp{2}, 'sigma');
threw = false; msg = '';
try; buildExpTens(pIO, [], 'specs', sp, 'verbose', false); catch e; threw = true; msg = e.message; end
results{end+1,1} = 'preMaetIo: missing refused by name'; %#ok<SAGROW>
results{end,2}   = threw && ~isempty(strfind(msg, 'No sigma for attribute ''time''')); %#ok<STREMP>

sp = mkSpecs(); sp{1}.sigma = NaN;
threw = false; msg = '';
try; buildExpTens(pIO, [], 'specs', sp, 'verbose', false); catch e; threw = true; msg = e.message; end
results{end+1,1} = 'preMaetIo: NA refused and says why'; %#ok<SAGROW>
results{end,2}   = threw && ~isempty(strfind(msg, 'sigma for attribute ''pitch'' is NA')); %#ok<STREMP>

d = buildExpTens(pIO, [], 'specs', sp, 'sigma', [0.5 0.25], 'verbose', false);
results{end+1,1} = 'preMaetIo: NA recoverable at the call'; %#ok<SAGROW>
results{end,2}   = abs(d.sigma(1) - 0.5) < 1e-12;

% Only sigma and isPer are compulsory: a period is inert on a
% non-periodic attribute.
sp = mkSpecs(); sp{2} = rmfield(sp{2}, 'period');
d = buildExpTens(pIO, [], 'specs', sp, 'verbose', false);
results{end+1,1} = 'preMaetIo: period defaults to zero'; %#ok<SAGROW>
results{end,2}   = d.period(2) == 0;

% A spec written for either language reads in both.
sp = mkSpecs(); sp{1} = rmfield(sp{1}, 'isPer'); sp{1}.is_per = true;
d = buildExpTens(pIO, [], 'specs', sp, 'verbose', false);
results{end+1,1} = 'preMaetIo: snake_case alias is read'; %#ok<SAGROW>
results{end,2}   = d.isPer(1) == 1;

% ---- Operator rules ----

spD = {struct('name','p','r',1,'rel',false,'sym',true, ...
              'sigma',10,'isPer',false,'period',0)};
[~,~,s1] = unpackPreMaet(differenceEvents({[1 2 3 4]}, [], 1, 'specs', spD));
[~,~,s2] = unpackPreMaet(differenceEvents({[1 2 3 4]}, [], 2, 'specs', spD));
results{end+1,1} = 'preMaetIo: difference scales sigma by sqrt(C(2k,k))'; %#ok<SAGROW>
results{end,2}   = abs(s1{1}.sigma - 10*sqrt(2)) < 1e-9 && ...
                   abs(s2{1}.sigma - 10*sqrt(6)) < 1e-9;

% A covariance is in squared units, so it takes C(2k, k) where a width
% takes its root.
C = diag([0.04 0.09 0.16]);
spC = {struct('name','t','r',1,'rel',false,'sym',true, ...
              'sigma',C,'isPer',false,'period',0)};
[~,~,sC] = unpackPreMaet(differenceEvents({[1 2 3 4]}, [], 1, 'specs', spC));
results{end+1,1} = 'preMaetIo: difference scales a covariance by C(2k,k)'; %#ok<SAGROW>
results{end,2}   = max(max(abs(sC{1}.sigma - C * 2))) < 1e-12;

spB = mkSpecs(); spB = spB(1);
[~,~,sb] = unpackPreMaet(bindEvents({[60 62 64 65]}, [], 2, 'specs', spB));
[~,~,st] = unpackPreMaet(translateAttributes({[60 62 64 65]}, [], {5}, 'specs', spB));
[~,~,sw] = unpackPreMaet(weightEvents({[60 62 64 65]}, [], 1, 1, 62, 0, 'sd', 2, ...
                        'dropInputAttr', false, 'specs', spB));
results{end+1,1} = 'preMaetIo: bind/translate/weight carry geometry'; %#ok<SAGROW>
results{end,2}   = sb{1}.sigma == 0.5 && st{1}.sigma == 0.5 && ...
                   sw{1}.sigma == 0.5 && sb{1}.period == 12;

[~,~,sc] = unpackPreMaet(transformAttributes({[60 62 64]}, [], {{'midi','cents'}}, 'specs', spB));
[~,~,sa] = unpackPreMaet(transformAttributes({[60 62 64]}, [], {{'affine','scale',3}}, 'specs', spB));
results{end+1,1} = 'preMaetIo: affine maps scale sigma and period'; %#ok<SAGROW>
results{end,2}   = abs(sc{1}.sigma - 50) < 1e-12 && ...
                   abs(sc{1}.period - 1200) < 1e-12 && ...
                   abs(sa{1}.sigma - 1.5) < 1e-12 && ...
                   abs(sa{1}.period - 36) < 1e-12;

% A non-linear map carries no width across. A width is still meaningful
% in the new coordinate -- after a log it expresses a ratio rather than a
% difference -- but the local scaling varies across the range, so no
% single value is the image of the old one, and NA marks the absence of a
% canonical choice.
naOk = true;
for tr = {{{'midi','hz'}}, {'log'}}
    [~,~,sn] = unpackPreMaet(transformAttributes({[60 62 64]}, [], tr{1}, 'specs', spB));
    naOk = naOk && isnan(sn{1}.sigma) && isnan(sn{1}.period);
end
results{end+1,1} = 'preMaetIo: non-linear maps leave NA'; %#ok<SAGROW>
results{end,2}   = naOk;

% ---- CSV round trip ----

FLAT = ['name,sigma,r,rel,per,P,sym,n = 1,n = 2,n = 3' sprintf('\n') ...
    'pitch,0.5,2,0,1,12,1,"{60, 64, 67}","{62, 65, 69}","{60, 64, 67}"' sprintf('\n') ...
    'onset,0.25,1,0,0,,1,0,1,2' sprintf('\n')];
NESTED = ['name,sigma,r,rel,per,P,sym,n = 1,n = 2' sprintf('\n') ...
    'pitch,0.15,"(1, 3)","(0, 1)",1,12,"(1, 0)","({60, 64, 67}, {62, 67, 71}, {60, 64, 67})","({62, 65, 69}, {55, 59, 62}, {60, 64, 67})"' sprintf('\n') ...
    'metre,0.1,1,0,0,,1,1^(1),0.5^(0.5)' sprintf('\n')];
COV = ['name,sigma,r,rel,per,P,sym,n = 1,n = 2' sprintf('\n') ...
    'trigram,"cov(sd_position=0.2, sd_interval=0.3, sd_shift=0.5)",3,0,0,,0,"(60, 62, 64)","(62, 64, 65)"' sprintf('\n')];

srcs = {FLAT, NESTED, COV};
labels = {'flat', 'nested', 'covariance'};
for i = 1:3
    [p, w, sp2] = unpackPreMaet(readPreMaet(srcs{i}));
    results{end+1,1} = sprintf('preMaetIo: %s round trip is byte-exact', labels{i}); %#ok<SAGROW>
    results{end,2}   = strcmp(writePreMaet([], p, w, sp2), srcs{i});
end

RAGGED = ['name,sigma,r,rel,per,P,sym,a,b,c' sprintf('\n') ...
    'pitch,0.5,1,0,0,,1,"{60, 64, 67}","{62, 65}","{60, 64, 67, 71}"' sprintf('\n')];
[pr, wr, spr] = unpackPreMaet(readPreMaet(RAGGED));
results{end+1,1} = 'preMaetIo: ragged events pad and unpad'; %#ok<SAGROW>
results{end,2}   = isequal(size(pr{1}), [4 3]) && isnan(pr{1}(4,1)) && ...
                   isnan(pr{1}(3,2)) && ...
                   strcmp(writePreMaet([], pr, wr, spr, ...
                          'headings', {'a','b','c'}), RAGGED);

% A file never writes tags down: the bracket structure is the level
% structure.
[~, ~, spn] = unpackPreMaet(readPreMaet(NESTED));
results{end+1,1} = 'preMaetIo: tags rebuilt from the brackets'; %#ok<SAGROW>
results{end,2}   = isfield(spn{1}, 'tags') && ...
                   isequal(spn{1}.tags(:)', [0 0 0 1 1 1 2 2 2]);

[pf, wf, spf] = unpackPreMaet(readPreMaet(FLAT));
df = buildExpTens(pf, wf, 'specs', spf, 'verbose', false);
results{end+1,1} = 'preMaetIo: builds from the file, nothing supplied'; %#ok<SAGROW>
results{end,2}   = df.dim == 3 && ...
                   abs(cosSimExpTens(df, df, 'verbose', false) - 1) < 1e-12;

% A file may be written by hand in a spreadsheet or by the Python twin;
% the parameter is the same either way.
camel = strrep(strrep(strrep(COV, 'sd_position', 'sdPosition'), ...
    'sd_interval', 'sdInterval'), 'sd_shift', 'sdShift');
[~, ~, spa] = unpackPreMaet(readPreMaet(COV));
[~, ~, spb] = unpackPreMaet(readPreMaet(camel));
results{end+1,1} = 'preMaetIo: either covariance spelling is read'; %#ok<SAGROW>
results{end,2}   = max(max(abs(spa{1}.sigma - spb{1}.sigma))) < 1e-12;

Cbad = [1 0.9 0.1; 0.9 1 0.2; 0.1 0.2 1];
[pc, wc, spc] = unpackPreMaet(readPreMaet(COV));
spc{1}.sigma = Cbad;
threw = false;
try; writePreMaet([], pc, wc, spc); catch; threw = true; end
results{end+1,1} = 'preMaetIo: covariance outside the family refused'; %#ok<SAGROW>
results{end,2}   = threw;

NA_SRC = ['name,sigma,r,rel,per,P,sym,n = 1' sprintf('\n') ...
    'pitch,NA,1,0,0,,1,60' sprintf('\n')];
[pn, wn, spNa] = unpackPreMaet(readPreMaet(NA_SRC));
results{end+1,1} = 'preMaetIo: NA survives the round trip'; %#ok<SAGROW>
results{end,2}   = isnan(spNa{1}.sigma) && ...
                   strcmp(writePreMaet([], pn, wn, spNa), NA_SRC);

threw = false;
try; readPreMaet(['a,b,c' sprintf('\n') '1,2,3' sprintf('\n')]); catch; threw = true; end
results{end+1,1} = 'preMaetIo: bad header is named'; %#ok<SAGROW>
results{end,2}   = threw;

% A file records the pre-MAET; it does not display it.
spWide = {struct('name','x','r',1,'rel',false,'sym',true, ...
                 'sigma',1,'isPer',false,'period',0)};
wide = writePreMaet([], {0:19}, [], spWide, 'maxEvents', 3);
results{end+1,1} = 'preMaetIo: csv elides nothing'; %#ok<SAGROW>
results{end,2}   = ~isempty(strfind(wide, 'n = 20')) && ...
                   isempty(strfind(wide, '...')); %#ok<STREMP>


if standalone
    nPass = 0; nFail = 0;
    for i = 1:size(results, 1)
        if results{i,2}
            nPass = nPass + 1;
            fprintf('  PASS  %s\n', results{i,1});
        else
            nFail = nFail + 1;
            fprintf('  FAIL  %s\n', results{i,1});
        end
    end
    fprintf('\n=== test_pre_maet_io: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_pre_maet_io:failed', '%d test(s) failed.', nFail);
    end
end
