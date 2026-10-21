function test_weight_search
% Fast numerical checks; no hardware or existing MATLAB workspace changes.
T = zeros(16,16,2);
T(5:10,5:6,1) = 1;
T(5:6,5:10,2) = 1;
[s,e] = hologram_quality(3*T,T);
assert(s < 1e-12 && all(e < 1e-12));
I = T + 0.01;
[s,e] = hologram_quality(I,T);
Tp = zeros(32,32,2); Ip = Tp;
Tp(1:16,1:16,:) = T; Ip(1:16,1:16,:) = I;
[sp,ep] = hologram_quality(Ip,Tp);
assert(abs(s-sp) < 1e-12 && max(abs(e-ep)) < 1e-12);

A = zeros(16); A(3:14,3:14) = 1;
f = (-8:7)/16; [fx,fy] = meshgrid(f,f);
H = cat(3, ifftshift(exp(-1i*3*(fx.^2+fy.^2))), ...
    ifftshift(exp(-1i*5*(fx.^2+fy.^2))));
o = struct('maxn',3,'seed',1,'coarse',[0.5 0.75], ...
    'fine_radius',0,'fine_step',0.025);
[U,r] = optimize_depth_weights(A,T,H,o);
[U2,r2] = optimize_depth_weights(A,T,H,o);
assert(norm(U(:)-U2(:)) < 1e-12 && r.score == r2.score);
assert(max(abs(abs(U(:))-A(:))) < 1e-12);
R = zeros(size(T));
for p=1:2
    R(:,:,p) = abs(ifft2(fft2(U).*H(:,:,p))).^2;
end
[actual,errs] = hologram_quality(R,T);
assert(abs(actual-r.score) < 1e-12 && max(abs(errs-r.errors)) < 1e-12);
assert(abs(min(r.table.Score)-r.score) < 1e-12);
assert(height(r.table)==2);
fprintf('PASS: scale invariance, zero padding, deterministic starts, amplitude constraint, selected-score consistency.\n');
end
