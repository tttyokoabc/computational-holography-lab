%% 1. 初始化与物理参数
clc;
clear;
close all;
rng(1);

lambda = 633e-9;       % 示例波长，单位m
dx = 8e-6;            % 像素间距，单位m
z0 = 50e-3;           % 设计重建距离：50 mm

slm_Ny = 1080;        % 导出画布：行数
slm_Nx = 1920;        % 导出画布：列数
Ny = 1536;           % 计算网格含外围留白，不直接导出
Nx = 2560;
N_active = 256;       % 中央被照明区域

%% 2. 空间坐标与SLM面入射振幅
x = (-Nx/2:Nx/2-1) * dx;
y = (-Ny/2:Ny/2-1) * dx;
x_mm = x * 1e3;
y_mm = y * 1e3;

% 中央256×256区域振幅为1，外围为0
A_slm = zeros(Ny, Nx);

active_y = (Ny-N_active)/2 + (1:N_active);
active_x = (Nx-N_active)/2 + (1:N_active);
slm_rows = (Ny-slm_Ny)/2 + (1:slm_Ny);
slm_cols = (Nx-slm_Nx)/2 + (1:slm_Nx);
% 外围无光是照明假设，不能通过相位灰度图实现遮光。
A_slm(active_y, active_x) = 1;

%% 3. 构造目标H
small_target = zeros(N_active, N_active);

small_target(65:192, 65:88) = 1;
small_target(65:192, 169:192) = 1;
small_target(117:140, 65:192) = 1;

target_intensity = zeros(Ny, Nx);
target_intensity(active_y, active_x) = small_target;

target_amp = sqrt(target_intensity);

% 使两面的约束具有相同总能量
input_energy = sum(A_slm(:).^2);
target_energy = sum(target_amp(:).^2);

target_amp = target_amp * sqrt(input_energy / target_energy);

% 用于评价图像形状的归一化目标强度
target_norm = target_intensity / sum(target_intensity(:));

%% 4. 空间频率网格与角谱传递函数
% 空间频率单位：每米，即cycles/m
fx = (-Nx/2:Nx/2-1) / (Nx*dx);
fy = (-Ny/2:Ny/2-1) / (Ny*dx);
[Fx, Fy] = meshgrid(fx, fy);

q = 1/lambda^2 - Fx.^2 - Fy.^2;

% 当前参数下，计算网格内的分量都是传播波
assert(all(q(:) > 0), ...
    '当前网格包含倏逝波，需要修改传播处理。');

% 纵向波数，单位rad/m
kz = 2*pi*sqrt(q);

% H先按中心化频率构造，再改成fft2的频率排列
H_forward = ifftshift(exp(1i*z0*kz));
H_backward = conj(H_forward);
%% 5. 两个深度的目标振幅

z_planes = [40 60] * 1e-3;       % H在40 mm，T在60 mm
maxn = 100;

% 第一个目标沿用前面创建的H
I_H = target_intensity;

% 第二个目标：T
small_T = zeros(N_active, N_active);
small_T(65:88, 65:192) = 1;       % 顶部横线
small_T(65:192, 117:140) = 1;     % 中央竖线

I_T = zeros(Ny, Nx);
I_T(active_y, active_x) = small_T;

target_I = cat(3, I_H, I_T);

A_targets = zeros(Ny, Nx, 2);
target_norms = zeros(Ny, Nx, 2);
H_planes = complex(zeros(Ny, Nx, 2));

input_energy = sum(A_slm(:).^2);

for p = 1:2
    I = target_I(:, :, p);

    % 两个目标面各自具有与输入相同的总能量
    % 它们是同一束光经过的不同截面，不是各分一半能量
    A_targets(:, :, p) = ...
        sqrt(I * input_energy / sum(I(:)));

    target_norms(:, :, p) = I / sum(I(:));

    H_planes(:, :, p) = ...
        ifftshift(exp(1i*z_planes(p)*kz));
end

%% 6. 自动搜索权重：固定评价函数、随机初相和迭代预算
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
options = struct('maxn', 100, 'seed', 1, ...
    'coarse', 0.1:0.1:0.9, 'fine_step', 0.025, 'fine_radius', 0.1);
tic;
[U_slm, search_result] = optimize_depth_weights(A_slm, target_I, H_planes, options);
elapsed_seconds = toc;
disp(sortrows(search_result.table, 'WeightH'));
fprintf('搜索所得权重 H:T = %.3f:%.3f，J = %.6f，迭代 %d\n', ...
    search_result.weights, search_result.score, search_result.best_iteration);

%% 7. 搜索曲线与最佳结果
summary = sortrows(search_result.table, 'WeightH');
figure('Name', '权重搜索');
plot(summary.WeightH, summary.RelativeErrorH, '-o', ...
     summary.WeightH, summary.RelativeErrorT, '-o', ...
     summary.WeightH, summary.Score, '--k');
xlabel('H的生成权重'); ylabel('相对强度误差');
legend('H', 'T', 'J：两层较大误差', 'Location', 'best'); grid on;
xline(search_result.weights(1), ':', '选定权重');

spectrum = fft2(U_slm);
I_best = zeros(Ny, Nx, 2);
for p = 1:2
    I_best(:,:,p) = abs(ifft2(spectrum .* H_planes(:,:,p))).^2;
end
figure('Name', '搜索得到的双平面重建');
clim_max = max(I_best(:));
for p = 1:2
    subplot(1,2,p);
    imagesc(x_mm, y_mm, I_best(:,:,p), [0 clim_max]);
    axis image; xlim([-1.2 1.2]); ylim([-1.2 1.2]);
    xlabel('x / mm'); ylabel('y / mm');
    title(sprintf('z = %.0f mm', z_planes(p)*1e3));
end
colormap gray;

%% 8. 保存可复现结果（原始相位，不在此进行设备标定）
out_dir = make_result_dir('gs_weight_search');
phase_active = angle(U_slm(active_y,active_x));
save(fullfile(out_dir, 'weight_search.mat'), 'U_slm', 'phase_active', ...
    'search_result', 'target_I', 'A_slm', 'lambda', 'dx', 'z_planes', ...
    'Ny', 'Nx', 'N_active', 'elapsed_seconds', '-v7.3');
writetable(summary, fullfile(out_dir, 'weight_search.csv'));
fprintf('结果保存到：\n%s\n', out_dir);
