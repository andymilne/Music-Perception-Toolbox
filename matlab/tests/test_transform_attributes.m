%% test_transform_attributes.m — transformAttributes tests
%
%  Named transforms, scale conversions (formerly convertPitch), user
%  functions, the sign attribute, and the domain refusals. Mirror of
%  Python tests/test_transform_attributes.py; the cross-language
%  reference block asserts the same numbers as the Python test.
%
%  Standalone-runnable; appends to `results` when called from
%  test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

tol = 1e-10;

% --- Scale conversions (bare-array form) -----------------------------

results{end+1,1} = 'transform: Hz->MIDI';
results{end,2}   = abs(transformAttributes(440, [], {'hz', 'midi'}) - 69) < tol;

results{end+1,1} = 'transform: MIDI->Hz';
results{end,2}   = abs(transformAttributes(60, [], {'midi', 'hz'}) - 261.6256) / 261.6256 < 1e-4;

results{end+1,1} = 'transform: Hz->cents';
results{end,2}   = abs(transformAttributes(440, [], {'hz', 'cents'}) - 6900) < tol;

results{end+1,1} = 'transform: octave is MIDI/12';
results{end,2}   = abs(transformAttributes(440, [], {'hz', 'octave'}) - 69/12) < tol ...
                   && abs(transformAttributes(5.75, [], {'octave', 'midi'}) - 69) < tol;

results{end+1,1} = 'transform: identity';
results{end,2}   = isequal(transformAttributes([100, 200, 300], [], {'hz', 'hz'}), [100, 200, 300]);

scales = {'midi', 'cents', 'octave', 'mel', 'bark', 'erb', 'greenwood'};
for i = 1:numel(scales)
    rt = transformAttributes(transformAttributes(440, [], {'hz', scales{i}}), [], {scales{i}, 'hz'});
    results{end+1,1} = ['transform: roundtrip ' scales{i}]; %#ok<SAGROW>
    results{end,2}   = abs(rt - 440) / 440 < 1e-8;
end

out = transformAttributes([261.63, 440, 880], [], {'hz', 'midi'});
results{end+1,1} = 'transform: vectorised, shape kept';
results{end,2}   = all(abs(out - [60, 69, 81]) < 0.01) && isequal(size(out), [1 3]);
out = transformAttributes([261.63; 440; 880], [], {'hz', 'midi'});
results{end,2}   = results{end,2} && isequal(size(out), [3 1]);
out = transformAttributes([261.63, 440; 880, 220], [], {'hz', 'midi'});
results{end,2}   = results{end,2} && isequal(size(out), [2 2]);

results{end+1,1} = 'transform: unknown scale errors';
results{end,2}   = throwsErrorWithId(@() transformAttributes(440, [], {'hz', 'bogus'}), ...
                                     'transformAttributes:unknownScale');

results{end+1,1} = 'transform: scale pair takes no parameters';
results{end,2}   = throwsErrorWithId(@() transformAttributes(440, [], struct('from', 'hz', 'to', 'midi', 'ref', 2)), ...
                                     'transformAttributes:badParameter');

results{end+1,1} = 'transform: unwrapped pair with A=2 gets a hint';
results{end,2}   = errorMessageContains(@() transformAttributes({440, 1}, [], {'hz', 'cents'}), ...
                                        'scale name');

% --- Named transforms --------------------------------------------------

results{end+1,1} = 'transform: log default, base, and offset';
results{end,2}   = abs(transformAttributes(exp(1), [], 'log') - 1) < tol ...
                   && abs(transformAttributes(8, [], {'log', 'base', 2}) - 3) < tol ...
                   && abs(transformAttributes(1000, [], struct('name', 'log', 'base', 10)) - 3) < tol ...
                   && transformAttributes(0, [], {'log', 'offset', 1}) == 0 ...
                   && abs(transformAttributes(7, [], {'log', 'base', 2, 'offset', 1}) - 3) < tol;

results{end+1,1} = 'transform: power and affine';
results{end,2}   = abs(transformAttributes(9, [], {'power', 'exponent', 0.5}) - 3) < tol ...
                   && abs(transformAttributes(2, [], {'power', 'exponent', 3}) - 8) < tol ...
                   && abs(transformAttributes(100, [], {'affine', 'scale', 0.01, 'offset', -1})) < tol ...
                   && transformAttributes(5, [], 'affine') == 5;

results{end+1,1} = 'transform: required and unknown parameters';
results{end,2}   = throwsErrorWithId(@() transformAttributes(2, [], 'power'), ...
                                     'transformAttributes:badParameter') ...
                   && throwsErrorWithId(@() transformAttributes(2, [], {'log', 'period', 3}), ...
                                        'transformAttributes:badParameter') ...
                   && throwsErrorWithId(@() transformAttributes(2, [], 'cube'), ...
                                        'transformAttributes:unknownTransform') ...
                   && throwsErrorWithId(@() transformAttributes(2, [], 'log1p'), ...
                                        'transformAttributes:unknownTransform');

results{end+1,1} = 'transform: names are case-insensitive';
results{end,2}   = abs(transformAttributes(8, [], {'LOG', 'Base', 2}) - 3) < tol ...
                   && abs(transformAttributes(440, [], {'Hz', 'MIDI'}) - 69) < tol;

% --- Domain refusals ---------------------------------------------------

results{end+1,1} = 'transform: zero under log refused with remedies';
results{end,2}   = throwsErrorWithId(@() transformAttributes({[0.5 0 0.25]}, [], {'log'}), ...
                                     'transformAttributes:domainZero') ...
                   && errorMessageContains(@() transformAttributes({[0.5 0 0.25]}, [], {'log'}), ...
                                           'event 2') ...
                   && errorMessageContains(@() transformAttributes({[0.5 0 0.25]}, [], {'log'}), ...
                                           'bindEvents') ...
                   && errorMessageContains(@() transformAttributes({[0.5 0 0.25]}, [], {'log'}), ...
                                           'offset');

[outLO, ~, ~] = unpackPreMaet(transformAttributes({[0 3]}, [], {{'log', 'offset', 1, 'base', 2}}));
results{end+1,1} = 'transform: log offset domain is x + offset';
results{end,2}   = max(abs(outLO{1} - [0 2])) < tol ...
                   && errorMessageContains(@() transformAttributes({[0.5 1]}, [], {{'log', 'offset', -0.5}}), ...
                                           'x + offset <= 0');

results{end+1,1} = 'transform: negative under log recommends sign';
results{end,2}   = throwsErrorWithId(@() transformAttributes({[1 -2]}, [], {'log'}), ...
                                     'transformAttributes:domainNegative') ...
                   && errorMessageContains(@() transformAttributes({[1 -2]}, [], {'log'}), ...
                                           '''sign'', true');

results{end+1,1} = 'transform: negative under a non-magnitude transform has no sign hint';
results{end,2}   = throwsErrorWithId(@() transformAttributes({-1}, [], {{'hz', 'midi'}}), ...
                                     'transformAttributes:domainNegative') ...
                   && ~errorMessageContains(@() transformAttributes({-1}, [], {{'hz', 'midi'}}), ...
                                            '''sign'', true');

results{end+1,1} = 'transform: zero Hz refused';
results{end,2}   = throwsErrorWithId(@() transformAttributes({0}, [], {{'hz', 'cents'}}), ...
                                     'transformAttributes:domainZero');

results{end+1,1} = 'transform: non-finite input refused';
results{end,2}   = throwsErrorWithId(@() transformAttributes({[1 Inf]}, [], {'affine'}), ...
                                     'transformAttributes:nonFiniteInput');

results{end+1,1} = 'transform: function handle must return finite, same shape';
results{end,2}   = throwsErrorWithId(@() transformAttributes({[1 2]}, [], {@(x) log(x - 1)}), ...
                                     'transformAttributes:nonFiniteOutput') ...
                   && throwsErrorWithId(@() transformAttributes({[1 2]}, [], {@(x) x(1)}), ...
                                        'transformAttributes:badOutputShape');

sp = flatSpecs({zeros(1, 2), zeros(1, 2)}, 'name', {'pitch', 'ioi'});
results{end+1,1} = 'transform: message names the attribute';
results{end,2}   = errorMessageContains(@() transformAttributes({[1 2], [0.5 0]}, [], ...
                                            {[], 'log'}, 'specs', sp), 'attribute 2 (''ioi'')');

% --- Cell form: threading, handles, sign attribute --------------------

p = {[1 2], [4 9]};
[out, w, sp] = unpackPreMaet(transformAttributes(p, [], {[], {'power', 'exponent', 0.5}}));
results{end+1,1} = 'transform: [] leaves attribute; weights/specs threaded';
results{end,2}   = isequal(out{1}, [1 2]) && max(abs(out{2} - [2 3])) < tol ...
                   && isempty(w) && numel(sp) == 2;
[out2, ~, ~] = unpackPreMaet(transformAttributes(p, [], {'power', 'exponent', 0.5}));
results{end+1,1} = 'transform: single entry broadcasts to all attributes';
results{end,2}   = max(abs(out2{1} - [1 sqrt(2)])) < tol;

spIn = flatSpecs({[1 2]}, 'r', 2, 'name', 'x');
[~, w, sp] = unpackPreMaet(transformAttributes({[1 2]}, {[0.5 0.5]}, {'log'}, 'specs', spIn));
results{end+1,1} = 'transform: weights and specs pass through';
results{end,2}   = isequal(sp{1}, spIn{1}) && isequal(w{1}, [0.5 0.5]);

[out, ~, ~] = unpackPreMaet(transformAttributes({[1 4; 9 16]}, [], {@sqrt}));
results{end+1,1} = 'transform: function handle on a K_total x N matrix';
results{end,2}   = max(abs(out{1}(:) - [1; 3; 2; 4])) < tol;

p = {[2 -3 0], [1 1 1]};
sp = flatSpecs(p, 'name', {'ivl', 't'});
[out, w, s] = unpackPreMaet(transformAttributes(p, {1, 2}, {{'log', 'offset', 1}, []}, 'specs', sp, 'sign', [true false]));
results{end+1,1} = 'transform: sign attribute inserted after its source';
results{end,2}   = numel(out) == 3 && numel(w) == 3 && numel(s) == 3 ...
                   && max(abs(out{1} - [log(3) log(4) 0])) < tol ...
                   && isequal(out{2}, 0.5 * [1 -1 0]) && isequal(out{3}, [1 1 1]) ...
                   && isequal(s{2}.r, 1) && isequal(s{2}.rel, false) && isequal(s{2}.sym, true) ...
                   && strcmp(s{2}.name, 'ivl_sign') ...
                   && isequal(w, {1, 1, 2});

spN = struct('tags', [0 1], 'r', [1 2], 'sym', [true true], 'rel', [0 1]);
[out, ~, s] = unpackPreMaet(transformAttributes({[1 -2; -3 4]}, [], {{'power', 'exponent', 0.5}}, 'specs', {spN}, 'sign', true));
results{end+1,1} = 'transform: sign on a nested spec clears rel';
results{end,2}   = isequal(out{2}, 0.5 * [1 -1; -1 1]) && isequal(s{2}.rel, [false false]) ...
                   && isequal(s{2}.tags, [0 1]) && strcmp(s{2}.name, 'sign');

[out, ~, ~] = unpackPreMaet(transformAttributes({[-4 9]}, [], {{'power', 'exponent', 0.5}}, 'sign', true));
results{end+1,1} = 'transform: sign with power';
results{end,2}   = max(abs(out{1} - [2 3])) < tol && isequal(out{2}, 0.5 * [-1 1]);

results{end+1,1} = 'transform: sign requires a magnitude transform';
results{end,2}   = throwsErrorWithId(@() transformAttributes({1}, [], {'affine'}, 'sign', true), ...
                                     'transformAttributes:signNotMagnitude') ...
                   && throwsErrorWithId(@() transformAttributes({1}, [], {[]}, 'sign', true), ...
                                        'transformAttributes:signNotMagnitude') ...
                   && throwsErrorWithId(@() transformAttributes({1}, [], {{'hz', 'midi'}}, 'sign', true), ...
                                        'transformAttributes:signNotMagnitude');

results{end+1,1} = 'transform: zero under log still refused with sign';
results{end,2}   = throwsErrorWithId(@() transformAttributes({[1 0]}, [], {'log'}, 'sign', true), ...
                                     'transformAttributes:domainZero');

results{end+1,1} = 'transform: bare form rejects cell arguments';
results{end,2}   = throwsErrorWithId(@() transformAttributes(1, {1}, 'log'), ...
                                     'transformAttributes:bareForm') ...
                   && throwsErrorWithId(@() transformAttributes(1, [], 'log', 'sign', true), ...
                                        'transformAttributes:bareForm');

results{end+1,1} = 'transform: length mismatches';
results{end,2}   = throwsErrorWithId(@() transformAttributes({1, 1}, [], 'log', 'sign', [true true true]), ...
                                        'transformAttributes:signLength') ...
                   && throwsErrorWithId(@() transformAttributes({1, [1 2]}, [], 'log'), ...
                                        'transformAttributes:eventCountMismatch');

% --- Composition with differencing and build ---------------------------

[p, w, sp] = unpackPreMaet(transformAttributes({[0.25 0.5 0.5 1]}, [], {{'log', 'base', 2}}));
[d, ~, ~] = unpackPreMaet(differenceEvents(p, w, 1, 'specs', sp));
results{end+1,1} = 'transform: log then difference gives log ratios';
results{end,2}   = max(abs(d{1} - [1 0 1])) < tol;

[d, w, sp] = unpackPreMaet(differenceEvents({[60 64 62 62 67]}, [], 1));
[p, w, sp] = unpackPreMaet(transformAttributes(d, w, {{'log', 'offset', 1}}, 'specs', sp, 'sign', true));
dens = buildExpTens(p, w, 'specs', sp, 'sigma', [0.2 0.3], 'isPer', [false false], ...
                    'period', [0 0], 'verbose', false);
results{end+1,1} = 'transform: difference then log(x+1) with sign feeds build';
results{end,2}   = numel(dens.sigma) == 2 ...
                   && abs(cosSimExpTens(dens, dens, 'verbose', false) - 1) < 1e-9;

% --- Cross-language reference values (shared with the Python test) ------

X = [0.25 0.5 1 2 4];
refT = { {'log', 'base', 2},        [-2 -1 0 1 2]; ...
         {'log', 'offset', 1},       [0.22314355131421 0.405465108108164 0.693147180559945 1.09861228866811 1.6094379124341]; ...
         {'power', 'exponent', 0.5}, [0.5 0.707106781186548 1 1.4142135623731 2]; ...
         {'power', 'exponent', 1.5}, [0.125 0.353553390593274 1 2.82842712474619 8]; ...
         {'affine', 'scale', 2, 'offset', -1}, [-0.5 0 1 3 7]; ...
         {'hz', 'erb'},              [0.0101480454786181 0.020285022354027 0.040525866692578 0.0808757881114107 0.16105386008722]; ...
         {'hz', 'octave'},           [-5.03135971352466 -4.03135971352466 -3.03135971352466 -2.03135971352466 -1.03135971352466] };
ok = true;
for i = 1:size(refT, 1)
    got = transformAttributes(X, [], refT{i, 1});
    ok = ok && max(abs(got - refT{i, 2})) <= 1e-9 * max(1, max(abs(refT{i, 2})));
end
results{end+1,1} = 'transform: cross-language reference table';
results{end,2}   = ok;

if standalone
    nPass = sum([results{:,2}]);
    nFail = size(results, 1) - nPass;
    for ii = 1:size(results, 1)
        if results{ii,2}
            fprintf('  PASS  %s\n', results{ii,1});
        else
            fprintf('  FAIL  %s\n', results{ii,1});
        end
    end
    fprintf('\n=== test_transform_attributes: %d passed, %d failed (of %d) ===\n\n', ...
            nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_transform_attributes:failed', '%d test(s) failed.', nFail);
    end
end
