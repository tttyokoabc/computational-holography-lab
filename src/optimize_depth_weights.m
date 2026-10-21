function [best_U, result] = optimize_depth_weights(A_slm, target_I, H_planes, options)
% Fixed-seed coarse/fine grid search for two-plane GS weights.
% Best means best evaluated weight/iteration under the declared objective.
if nargin < 4, options = struct(); end
if ~isfield(options, 'maxn'), options.maxn = 100; end
if ~isfield(options, 'seed'), options.seed = 1; end
if ~isfield(options, 'coarse'), options.coarse = 0.1:0.1:0.9; end
if ~isfield(options, 'fine_step'), options.fine_step = 0.025; end
if ~isfield(options, 'fine_radius'), options.fine_radius = 0.1; end
assert(size(target_I, 3) == 2 && isequal(size(target_I), size(H_planes)));
assert(isequal(size(A_slm), size(target_I(:, :, 1))));
assert(all(isfinite(A_slm(:))) && all(A_slm(:) >= 0) && any(A_slm(:) > 0));
assert(all(isfinite(target_I(:))) && all(target_I(:) >= 0));
assert(all(isfinite(H_planes(:))) && all(abs(abs(H_planes(:))-1) < 1e-10), ...
    'This solver requires unit-modulus propagation transfer functions.');
validateattributes(options.maxn, {'numeric'}, {'scalar','integer','positive'});
validateattributes(options.fine_step, {'numeric'}, {'scalar','real','finite','positive'});
validateattributes(options.fine_radius, {'numeric'}, {'scalar','real','finite','nonnegative'});
assert(~isempty(options.coarse) && all(isfinite(options.coarse)) && ...
    all(options.coarse >= 0 & options.coarse <= 1));

energy = sum(A_slm(:).^2);
A_targets = zeros(size(target_I));
for p = 1:2
    T = target_I(:, :, p);
    assert(sum(T(:)) > 0);
    A_targets(:, :, p) = sqrt(T * energy / sum(T(:)));
end

% Generate random phase only on illuminated pixels, so padding alone does
% not change their random samples. Every candidate starts from this U0.
stream = RandStream('mt19937ar', 'Seed', options.seed);
mask = A_slm > 0;
U0 = complex(zeros(size(A_slm)));
U0(mask) = A_slm(mask) .* exp(2i*pi*rand(stream, nnz(mask), 1));
best_score = inf;
best_U = U0;
rows = zeros(0, 5);
histories = {};
evaluated = [];

for stage = 1:2
    if stage == 1
        candidates = unique(options.coarse);
    else
        [~, index] = min(rows(:, 2));
        center = rows(index, 1);
        lo = max(0, center-options.fine_radius);
        hi = min(1, center+options.fine_radius);
        candidates = unique([lo:options.fine_step:hi, hi, center]);
    end
    for w = candidates(:).'
        if any(abs(evaluated-w) < 1e-10), continue; end
        evaluated(end+1) = w; %#ok<AGROW>
        U = U0;
        spectrum = fft2(U);
        history = zeros(options.maxn, 3);
        local_score = inf;
        local_iteration = 0;
        local_errors = [inf inf];
        for n = 1:options.maxn
            U_sum = complex(zeros(size(A_slm)));
            weights = [w, 1-w];
            for p = 1:2
                H = H_planes(:, :, p);
                V = ifft2(spectrum .* H);
                fixed = A_targets(:, :, p) .* exp(1i*angle(V));
                back = ifft2(fft2(fixed) .* conj(H));
                U_sum = U_sum + weights(p)*back;
            end
            U = A_slm .* exp(1i*angle(U_sum));
            spectrum = fft2(U);
            intensity = zeros(size(target_I));
            for p = 1:2
                intensity(:, :, p) = abs(ifft2(spectrum .* H_planes(:, :, p))).^2;
            end
            [score, errors] = hologram_quality(intensity, target_I);
            history(n, :) = [score, errors];
            if score < local_score
                local_score = score;
                local_iteration = n;
                local_errors = errors;
            end
            if score < best_score
                best_score = score;
                best_U = U;
                best_weight = w;
                best_iteration = n;
                best_errors = errors;
            end
            if mod(n, 20) == 0
                fprintf('wH=%.3f, iteration %d/%d, J=%.5f\n', w, n, options.maxn, score);
                drawnow;
            end
        end
        rows(end+1, :) = [w, local_score, local_errors, local_iteration]; %#ok<AGROW>
        histories{end+1} = history; %#ok<AGROW>
        fprintf('Completed wH=%.3f: best J=%.5f at iteration %d\n', ...
            w, local_score, local_iteration);
    end
end
result.table = array2table(rows, 'VariableNames', ...
    {'WeightH','Score','RelativeErrorH','RelativeErrorT','SelectedIteration'});
result.histories = histories; % Same ordering as table; columns J, eH, eT.
result.options = options;
result.weights = [best_weight, 1-best_weight];
result.best_iteration = best_iteration;
result.errors = best_errors;
result.score = best_score;
result.objective = 'max(relative L2 errors of energy-normalized intensities)';
result.boundary_warning = abs(best_weight-min(evaluated)) < 1e-10 || ...
    abs(best_weight-max(evaluated)) < 1e-10;
if result.boundary_warning
    warning('Best weight is at the searched boundary; consider expanding the search.');
end
end
