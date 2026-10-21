%% 1. 参数：先设计复场，再编码为纯相位（无GS迭代）
clear; close all; clc;
rng(7);
lambda = 633e-9;
dx = 8e-6;
N = 1536;                 % 含零填充的计算网格，避免外围光环回绕
N_active = 512;           % 均匀照明孔径；不是用相位图实现遮光
z_planes = [40 60]*1e-3;
weights = [1 1];          % 复场振幅权重
target_phase_mode = 'zero'; % 'random' 或 'zero'：比较目标相位的影响
n_frames = 10;            % 固定目标复场，只改变编码随机掩模
axicon_frequency = 40000; % cycles/m，小于像素Nyquist频率62500
run_name = 'random_two_depth_random';
if strcmp(target_phase_mode,'zero')
    run_name = 'random_two_depth_zero';
end
out_dir = make_result_dir(run_name);

%% 2. H/T目标强度与等能量目标复场
x = (-N/2:N/2-1)*dx;
[X,Y] = meshgrid(x,x);
active = (N-N_active)/2+(1:N_active);
pupil = zeros(N); pupil(active,active) = 1;
target = zeros(N,N,2);
c = (N-256)/2;
target(c+(65:192),c+(65:88),1) = 1;
target(c+(65:192),c+(169:192),1) = 1;
target(c+(117:140),c+(65:192),1) = 1;
target(c+(65:88),c+(65:192),2) = 1;
target(c+(65:192),c+(117:140),2) = 1;
U_target = complex(zeros(N,N,2));
for p = 1:2
    I = target(:,:,p);
    alpha = zeros(N);
    % 两种模式消耗相同随机数，使后续编码随机序列保持一致。
    target_random_phase = 2*pi*rand(nnz(I),1);
    if strcmp(target_phase_mode,'random')
        alpha(I>0) = target_random_phase;
    else
        assert(strcmp(target_phase_mode,'zero'), 'Unknown target phase mode.');
    end
    U_target(:,:,p) = sqrt(I/sum(I(:))).*exp(1i*alpha);
end

%% 3. 角谱传播与单目标回传验证
f = (-N/2:N/2-1)/(N*dx);
[Fx,Fy] = meshgrid(f,f);
q = 1/lambda^2-Fx.^2-Fy.^2;
assert(all(q(:)>0), 'This demonstration assumes propagating waves only.');
kz = 2*pi*sqrt(q);
H = complex(zeros(N,N,2));
V_parts = complex(zeros(N,N,2));
roundtrip_error = zeros(1,2);
for p = 1:2
    H(:,:,p) = ifftshift(exp(1i*z_planes(p)*kz));
    V_parts(:,:,p) = ifft2(fft2(U_target(:,:,p)).*conj(H(:,:,p)));
    recovered = ifft2(fft2(V_parts(:,:,p)).*H(:,:,p));
    difference = recovered-U_target(:,:,p);
    roundtrip_error(p) = norm(difference(:))/norm(reshape(U_target(:,:,p),[],1));
end
assert(max(roundtrip_error)<1e-10, 'Forward/backward propagation failed.');

%% 4. 复数相加：V=w1*V1+w2*V2；再考虑有限照明孔径
V = weights(1)*V_parts(:,:,1)+weights(2)*V_parts(:,:,2);
retained_energy = sum(abs(V(pupil==1)).^2)/sum(abs(V(:)).^2);
Amax = max(abs(V(pupil==1)));
T_full = V/Amax;
T = pupil.*T_full;
A = abs(T);
phi = angle(V);
fprintf('Round-trip errors: %.3g, %.3g\n',roundtrip_error);
fprintf('Complex-field energy retained by aperture: %.2f%%\n',100*retained_energy);

%% 5. 随机编码：以振幅A为概率选择目标相位，否则选择发散相位
% 在此传播符号约定下，正径向相位梯度对应向外传播。
psi = mod(2*pi*axicon_frequency*hypot(X,Y),2*pi);
F = exp(1i*phi);
D = exp(1i*psi);
I_mean = zeros(N,N,2);
for k = 1:n_frames
    R = false(N);
    R(active,active) = rand(N_active)<A(active,active);
    M = pupil.*(R.*F+(1-R).*D);
    if k==1
        M_first = M;
        phase_first = mod(angle(M),2*pi);
        I_first = reconstruct(M,H);
    end
    I_mean = I_mean+reconstruct(M,H)/n_frames;
end
assert(max(abs(abs(M(pupil==1))-1))<1e-12, 'Mask is not phase only.');

%% 6. 对照：同一目标复场与同一孔径；平均的是强度，不是相位
I_full = reconstruct(T_full,H);
I_complex = reconstruct(T,H);
I_phase = reconstruct(pupil.*F,H);
results = cat(4,I_full,I_complex,I_phase,I_first,I_mean);
method_names = {'Ideal complex (no aperture)','Complex + aperture', ...
    'Phase only','Random single frame','Random mean (10 frames)'};
method_names{5} = sprintf('Random mean (%d frames)',n_frames);
roi = abs(X)<=1.15e-3 & abs(Y)<=1.15e-3;
metrics = zeros(10,5);
labels = cell(10,1);
for m = 1:5
    for p = 1:2
        I = results(:,:,p,m); B = target(:,:,p);
        % ROI内最优强度比例拟合后的相对误差：关注形状，不掩盖效率。
        a = I(roi); b = B(roi);
        scale = dot(a,b)/dot(a,a);
        shape_error = norm(scale*a-b)/norm(b);
        fraction = sum(I(B>0))/sum(I(:));
        edge = true(N); edge(33:end-32,33:end-32) = false;
        edge_fraction = sum(I(edge))/sum(I(:));
        row = (m-1)*2+p;
        labels{row} = method_names{m};
        metrics(row,:) = [z_planes(p)*1e3,shape_error,fraction, ...
            sum(I(B>0))/sum(pupil(:)),edge_fraction];
    end
end
report = array2table(metrics,'VariableNames', ...
    {'Depth_mm','ROI_relative_error','Target_energy_fraction', ...
    'Target_energy_over_uniform_input','Boundary_energy_fraction'});
report = addvars(report,labels,'Before',1,'NewVariableNames','Method');
writetable(report,fullfile(out_dir,'metrics.csv'));
disp(report);
energy_error = abs(sum(I_first(:,:,1),'all')-sum(pupil(:)))/sum(pupil(:));
assert(energy_error<1e-10,'Propagation did not conserve energy.');
if max(metrics(:,5))>0.01
    warning('More than 1 percent energy near grid boundary; enlarge N to check wraparound.');
end

%% 7. 重建图：每种方法的两个平面共用同一亮度标尺
fig = figure('Name','Random complex encoding: two depths','Color','w', ...
    'Position',[50 50 1500 650]);
tiledlayout(2,6,'TileSpacing','compact','Padding','compact');
for p = 1:2
    nexttile((p-1)*6+1);
    imagesc(x*1e3,x*1e3,target(:,:,p),[0 1]);
    axis image; xlim([-1.15 1.15]); ylim([-1.15 1.15]);
    title(sprintf('Target: %.0f mm',z_planes(p)*1e3));
    xlabel('x (mm)'); ylabel('y (mm)');
    for m = 1:5
        common_peak = 0;
        for j = 1:2
            image_j = results(:,:,j,m);
            common_peak = max(common_peak,max(image_j(roi)));
        end
        nexttile((p-1)*6+m+1);
        imagesc(x*1e3,x*1e3,results(:,:,p,m),[0 common_peak]);
        axis image; xlim([-1.15 1.15]); ylim([-1.15 1.15]);
        title(method_names{m}); xlabel('x (mm)');
    end
end
colormap gray;
exportgraphics(fig,fullfile(out_dir,'reconstruction_comparison.png'),'Resolution',140);
fig2 = figure('Name','Encoding and redirected light','Color','w','Position',[60 60 1200 380]);
tiledlayout(1,3);
nexttile; imagesc(x*1e3,x*1e3,A); axis image; colorbar; title('Desired amplitude A');
nexttile; imagesc(x(active)*1e3,x(active)*1e3,phase_first(active,active));
axis image; colorbar; title('Encoded phase (radians)');
nexttile; imagesc(x*1e3,x*1e3,log10(I_first(:,:,1)+1e-8));
axis image; colorbar; title('40 mm full field: log10 intensity');
exportgraphics(fig2,fullfile(out_dir,'encoding_and_full_field.png'),'Resolution',140);
% 此图仅为512x512有效区仿真相位；上机前需要器件LUT和照明参数。
imwrite(uint8(floor(phase_first(active,active)/(2*pi)*256)), ...
    fullfile(out_dir,'phase_active_8bit.png'));
save(fullfile(out_dir,'simulation.mat'),'lambda','dx','N','N_active','z_planes', ...
    'weights','target_phase_mode','n_frames','axicon_frequency','target', ...
    'A','phi','psi','phase_first','results','report','roundtrip_error', ...
    'retained_energy','energy_error','-v7.3');
fprintf('Results saved to: %s\n',out_dir);

function I = reconstruct(U,H)
    spectrum = fft2(U);
    I = zeros(size(H));
    for p = 1:size(H,3)
        I(:,:,p) = abs(ifft2(spectrum.*H(:,:,p))).^2;
    end
end
