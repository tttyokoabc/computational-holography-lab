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

%% 6. 初始化同一张SLM相位图

rng(1);
U_slm = A_slm .* exp(1i*2*pi*rand(Ny, Nx));

error_history = zeros(maxn, 2);

% 保存综合误差最低的一轮，避免只依赖最后一次迭代
best_score = inf;
best_iteration = 0;
best_U_slm = U_slm;
weights = [0.75, 0.25];     % 顺序：H、T
%% 7. 双平面迭代

for n = 1:maxn

    U_sum = complex(zeros(Ny, Nx));
    spectrum_slm = fft2(U_slm);

    for p = 1:2
        H_p = H_planes(:, :, p);

        % 从同一个SLM光场传播到当前目标面
        U_p = ifft2(spectrum_slm .* H_p);

        % 替换为当前平面的目标振幅
        U_p_fixed = A_targets(:, :, p) .* ...
            exp(1i*angle(U_p));

        % 反传回SLM面，保留完整复数光场
        U_back = ifft2(fft2(U_p_fixed) .* conj(H_p));

        U_sum = U_sum + weights(p) * U_back;
    end

    % 按 weights 加权叠加，然后施加SLM振幅约束
    U_slm = A_slm .* exp(1i*angle(U_sum));

    % 用更新后的同一张全息图评价两个平面
    spectrum_slm = fft2(U_slm);

    for p = 1:2
        U_p = ifft2(spectrum_slm .* H_planes(:, :, p));
        I_p = abs(U_p).^2;
        I_p_norm = I_p / sum(I_p(:));

        difference = I_p_norm - target_norms(:, :, p);
        error_history(n, p) = ...
            sqrt(mean(difference(:).^2));
    end

    score = mean(error_history(n, :));

    if score < best_score
        best_score = score;
        best_iteration = n;
        best_U_slm = U_slm;
    end

    if mod(n, 20) == 0
        fprintf('双平面迭代：%d / %d\n', n, maxn);
    end
end

% 后续统一使用综合误差最低的那张全息图
U_slm = best_U_slm;

fprintf('采用综合误差最低的第 %d 次迭代结果。\n', ...
    best_iteration);

%% 8. 固定最终全息图，观察40、50、60 mm

z_observe = [40 50 60] * 1e-3;
I_observe = zeros(Ny, Nx, numel(z_observe));

spectrum_slm = fft2(U_slm);

for j = 1:numel(z_observe)
    H_z = ifftshift(exp(1i*z_observe(j)*kz));
    U_z = ifft2(spectrum_slm .* H_z);
    I_observe(:, :, j) = abs(U_z).^2;
end

figure('Name', '双深度目标与重建');

subplot(2, 3, 1);
imagesc(x_mm, y_mm, I_H, [0 1]);
axis image;
xlim([-1.2 1.2]);
ylim([-1.2 1.2]);
title('40 mm目标：H');

subplot(2, 3, 2);
axis off;
text(0.5, 0.5, '50 mm未设置目标', ...
    'HorizontalAlignment', 'center');

subplot(2, 3, 3);
imagesc(x_mm, y_mm, I_T, [0 1]);
axis image;
xlim([-1.2 1.2]);
ylim([-1.2 1.2]);
title('60 mm目标：T');

% 所有重建图共用一个显示范围
display_max = max(I_observe(:));

for j = 1:numel(z_observe)
    subplot(2, 3, j+3);

    imagesc(x_mm, y_mm, I_observe(:, :, j), ...
        [0 display_max]);

    axis image;
    xlim([-1.2 1.2]);
    ylim([-1.2 1.2]);

    xlabel('x / mm');
    ylabel('y / mm');
    title(sprintf('实际重建：%.0f mm', z_observe(j)*1e3));
end

colormap gray;

%% 9. 两个目标面的迭代误差

figure('Name', '双平面迭代误差');

plot(1:maxn, error_history(:, 1), ...
     1:maxn, error_history(:, 2));

grid on;
xlabel('迭代次数');
ylabel('归一化强度 RMSE');
legend('40 mm：H', '60 mm：T', 'Location', 'best');
title('同一张全息图对应的两个目标面');

%% 10. 最终相位图

phase_active = angle(U_slm(active_y, active_x));

figure('Name', '双深度相位全息图');
imagesc(x_mm(active_x), y_mm(active_y), phase_active, [-pi pi]);
axis image;
colormap gray;
colorbar;
xlabel('x / mm');
ylabel('y / mm');
title('同时用于两个深度的SLM相位 / rad');
%% 11. 保存双平面加权实验

out_dir = make_result_dir('gs_two_depth_weighted');

selected_errors = error_history(best_iteration, :);

save(fullfile(out_dir, 'two_depth_weighted.mat'), ...
    'U_slm', 'phase_active', ...
    'target_I', 'A_slm', ...
    'lambda', 'dx', 'Ny', 'Nx', 'slm_Ny', 'slm_Nx', 'N_active', ...
    'z_planes', 'z_observe', 'I_observe', ...
    'weights', 'maxn', 'error_history', ...
    'best_iteration', 'selected_errors');

summary_table = table( ...
    z_planes(:)*1e3, weights(:), selected_errors(:), ...
    'VariableNames', {'Depth_mm', 'Weight', 'SelectedRMSE'});

writetable(summary_table, ...
    fullfile(out_dir, 'selected_metrics.csv'));

disp(summary_table);
fprintf('实验数据保存到：\n%s\n', out_dir);
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
imwrite(I_T(slm_rows, slm_cols), fullfile(out_dir, 'target_T.png'));
fprintf('原生相位PNG已保存：\n%s\n', phase_file);
