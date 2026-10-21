function [score, errors] = hologram_quality(intensity, target)
% Relative L2 error of energy-normalized intensities; lower is better.
% The minimax score gives both planes equal importance, independently of
% the weights used to generate the hologram.
assert(isequal(size(intensity), size(target)), 'Array sizes must match.');
assert(all(isfinite(intensity(:))) && all(intensity(:) >= 0), ...
    'Reconstructed intensity must be finite and nonnegative.');
assert(all(isfinite(target(:))) && all(target(:) >= 0), ...
    'Target intensity must be finite and nonnegative.');
errors = zeros(1, size(target, 3));
for p = 1:size(target, 3)
    I = intensity(:, :, p);
    T = target(:, :, p);
    assert(sum(I(:)) > 0 && sum(T(:)) > 0, 'Each plane needs nonzero energy.');
    I = I / sum(I(:));
    T = T / sum(T(:));
    errors(p) = norm(I(:) - T(:)) / norm(T(:));
end
score = max(errors);
end
