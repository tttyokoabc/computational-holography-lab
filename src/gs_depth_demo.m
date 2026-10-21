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
maxn = 80;

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

%% 5. GS迭代：指定距离的相位全息图
U_target = target_amp .* exp(1i*2*pi*rand(Ny, Nx));

rmse_history = zeros(maxn, 1);

for n = 1:maxn

    % 目标面振幅约束
    U_target_fixed = target_amp .* exp(1i*angle(U_target));

    % 从目标面反传到SLM面
    U_back = ifft2(fft2(U_target_fixed) .* H_backward);

    % SLM面约束：被照明区域振幅为1，外围为0
    U_slm = A_slm .* exp(1i*angle(U_back));

    % 从SLM正传到设计距离z0
    U_target = ifft2(fft2(U_slm) .* H_forward);

    % 评价实际传播结果，而不是刚强制设置的目标
    I = abs(U_target).^2;
    I_norm = I / sum(I(:));
    difference = I_norm - target_norm;

    rmse_history(n) = sqrt(mean(difference(:).^2));

    if mod(n, 20) == 0
        fprintf('已完成 %d / %d 次迭代\n', n, maxn);
    end
end

%% 6. 固定全息图，观察不同距离
z_list = [40 50 60] * 1e-3;
I_stack = zeros(Ny, Nx, numel(z_list));
rmse_depth = zeros(numel(z_list), 1);

% 所有距离使用同一个SLM光场
spectrum_slm = fft2(U_slm);

for k = 1:numel(z_list)

    H_z = ifftshift(exp(1i*z_list(k)*kz));
    U_z = ifft2(spectrum_slm .* H_z);

    I_stack(:, :, k) = abs(U_z).^2;

    I = I_stack(:, :, k);
    I_norm = I / sum(I(:));
    difference = I_norm - target_norm;

    rmse_depth(k) = sqrt(mean(difference(:).^2));
end

%% 7. 显示目标与迭代误差
figure('Name', '目标与迭代误差');

subplot(1, 2, 1);
imagesc(x_mm, y_mm, target_intensity);
axis image;
xlim([-1.2 1.2]);
ylim([-1.2 1.2]);
colormap gray;
xlabel('x / mm');
ylabel('y / mm');
title('目标强度');

subplot(1, 2, 2);
plot(1:maxn, rmse_history);
grid on;
xlabel('迭代次数');
ylabel('归一化强度 RMSE');
title('设计平面的误差');

%% 8. 显示被照明区域的相位
phase_active = angle(U_slm(active_y, active_x));

figure('Name', '被照明区域的相位');
imagesc(x_mm(active_x), y_mm(active_y), phase_active, [-pi pi]);
axis image;
colormap gray;
colorbar;
xlabel('x / mm');
ylabel('y / mm');
title('SLM相位 / rad');

%% 9. 比较不同观察距离
% 三张图使用同一显示尺度
display_max = max(I_stack(:));

figure('Name', '同一张全息图：不同观察距离');

for k = 1:numel(z_list)
    subplot(1, 3, k);

    imagesc(x_mm, y_mm, I_stack(:, :, k), ...
        [0 display_max]);

    axis image;
    xlim([-1.2 1.2]);
    ylim([-1.2 1.2]);

    xlabel('x / mm');
    ylabel('y / mm');
    title(sprintf('z = %.0f mm', z_list(k)*1e3));
end
colormap gray;

depth_results = table(z_list(:)*1e3, rmse_depth, ...
    'VariableNames', {'Distance_mm', 'NormalizedIntensityRMSE'});

disp(depth_results);
%% 10. 固定全息图，扫描观察深度

z_scan = (35:1:65) * 1e-3;      % 35～65 mm，每隔1 mm
depth_error = zeros(size(z_scan));

% 同一张全息图的频谱，只计算一次
spectrum_slm = fft2(U_slm);

for j = 1:numel(z_scan)

    H_scan = ifftshift(exp(1i*z_scan(j)*kz));

    U_scan = ifft2(spectrum_slm .* H_scan);
    I_scan = abs(U_scan).^2;

    I_scan_norm = I_scan / sum(I_scan(:));
    difference = I_scan_norm - target_norm;

    depth_error(j) = sqrt(mean(difference(:).^2));
end

[min_error, best_idx] = min(depth_error);
best_z_mm = z_scan(best_idx) * 1e3;

figure('Name', '固定全息图的深度响应');
plot(z_scan*1e3, depth_error, '-o');
hold on;
xline(z0*1e3, '--r', '设计距离');
hold off;
grid on;

xlabel('观察距离 z / mm');
ylabel('归一化强度 RMSE');
title('同一张全息图在不同深度的重建误差');

fprintf('扫描范围内误差最低的距离：%.1f mm\n', best_z_mm);
fprintf('该距离的归一化强度RMSE：%.6g\n', min_error);
%% 保存本次深度实验
out_dir = make_result_dir('gs_depth_demo');

%% 导出SLM原生画布：只裁计算留白，不缩放相位
phase_slm = angle(U_slm(slm_rows, slm_cols));
phase_uint8 = uint8((phase_slm + pi) / (2*pi) * 255);
phase_file = fullfile(out_dir, 'phase_8bit_1920x1080.png');
imwrite(phase_uint8, phase_file);
info = imfinfo(phase_file);
assert(info.Height == slm_Ny && info.Width == slm_Nx && info.BitDepth == 8);
assert(isequal(imread(phase_file), phase_uint8));
illumination_slm = A_slm(slm_rows, slm_cols);
save(fullfile(out_dir, 'slm_export.mat'), 'phase_slm', 'illumination_slm', ...
    'lambda', 'dx', 'slm_Ny', 'slm_Nx', 'N_active');
% 这是照明示意，不能作为相位图加载来遮光
imwrite(illumination_slm, fullfile(out_dir, 'illumination_reference_only.png'));
imwrite(target_intensity(slm_rows, slm_cols), fullfile(out_dir, 'target_H.png'));
fprintf('原生相位PNG已保存：\n%s\n', phase_file);
