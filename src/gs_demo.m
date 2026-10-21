%% 1. 初始化
clc;
clear;
close all;

rng(1);                 % 固定随机起点，方便重复比较
Ny = 1080;               % 行数：SLM高度
Nx = 1920;               % 列数：SLM宽度
maxn = 200;              % GS迭代次数

%% 2. 创建目标振幅：黑底白色字母H
% 保持H的宽高比例，将原256像素模板放大到512像素后居中
small_H = zeros(256, 256);
small_H(65:192, 65:88) = 1;
small_H(65:192, 169:192) = 1;
small_H(117:140, 65:192) = 1;
small_H = repelem(small_H, 2, 2);
target_amp = zeros(Ny, Nx);
rows = (Ny-512)/2 + (1:512);
cols = (Nx-512)/2 + (1:512);
target_amp(rows, cols) = small_H;

figure('Name', '目标图像');
imagesc(target_amp, [0 1]);
axis image;
colormap gray;
colorbar;
title('目标振幅：H');

%% 3. 初始化成像面光场
random_phase = 2*pi*rand(Ny, Nx);
im_image = target_amp .* exp(1i*random_phase);

% 记录这些迭代次数对应的重建结果
record_steps = [1 2 5 20 50 100 200];
recon_intensity = zeros(Ny, Nx, numel(record_steps));

%% 4. GS迭代
for n = 1:maxn

    % 目标面：使用目标振幅，保留当前相位
    target_field = target_amp .* exp(1i*angle(im_image));

    % 反向传播至SLM面
    im_hologram = ifft2(ifftshift(target_field));

    % SLM面：振幅置为1，保留相位
    slm_field = exp(1i*angle(im_hologram));

    % 正向传播至成像面
    im_image = fftshift(fft2(slm_field));

    % 保存指定迭代次数下的重建强度
    idx = find(record_steps == n);
    if ~isempty(idx)
        recon_intensity(:, :, idx) = abs(im_image).^2;
    end
end

%% 5. 显示最终的相位全息图
phase_hologram = angle(im_hologram);

figure('Name', '相位全息图');
imagesc(phase_hologram, [-pi pi]);
axis image;
colormap gray;
colorbar;
title('SLM相位分布 / rad');

%% 6. 比较不同迭代次数的重建强度
% 所有重建图使用相同的显示范围，方便公平比较
display_max = max(recon_intensity(:));

figure('Name', 'GS迭代过程');
for k = 1:numel(record_steps)
    subplot(2, 4, k);
    imagesc(recon_intensity(:, :, k), [0 display_max]);
    axis image;
    title(sprintf('迭代 %d 次', record_steps(k)));
end
colormap gray;
%% 7. 定量评价已保存的重建结果

target_intensity = target_amp.^2;
target_norm = target_intensity / sum(target_intensity(:));

% 逻辑矩阵：H内部为true，背景为false
mask = target_intensity > 0;

K = numel(record_steps);
rmse_value = zeros(K, 1);
cv_value = zeros(K, 1);
energy_fraction = zeros(K, 1);

for k = 1:K
    I = recon_intensity(:, :, k);

    % 1. 总强度归一化后的RMSE
    I_norm = I / sum(I(:));
    difference = I_norm - target_norm;
    rmse_value(k) = sqrt(mean(difference(:).^2));

    % 2. H内部强度变异系数：标准差 / 平均值
    inside = I(mask);
    cv_value(k) = std(inside, 1) / mean(inside);

    % 3. 落在H内部的能量占比
    energy_fraction(k) = sum(inside) / sum(I(:));
end

% 在命令行显示结果表
results = table(record_steps(:), rmse_value, cv_value, ...
    100*energy_fraction, ...
    'VariableNames', {'Iteration', 'NormalizedIntensityRMSE', ...
    'IntensityCV', 'TargetEnergyPercent'});

disp(results);

% 绘制四个采样点的变化
figure('Name', '重建质量评价');

subplot(1, 3, 1);
plot(record_steps, rmse_value, '-o');
xlabel('迭代次数');
ylabel('归一化强度 RMSE');
grid on;

subplot(1, 3, 2);
plot(record_steps, cv_value, '-o');
xlabel('迭代次数');
ylabel('H内部强度变异系数');
grid on;

subplot(1, 3, 3);
plot(record_steps, 100*energy_fraction, '-o');
xlabel('迭代次数');
ylabel('目标区域能量占比 / %');
grid on;
%% 8. 保存原生分辨率结果
out_dir = make_result_dir('gs_demo');
save(fullfile(out_dir, 'gs_result.mat'), 'target_intensity', ...
    'phase_hologram', 'record_steps', 'rmse_value', 'Ny', 'Nx', 'maxn');
writetable(results, fullfile(out_dir, 'metrics.csv'));
imwrite(target_intensity, fullfile(out_dir, 'target.png'));
phase_uint8 = uint8((phase_hologram + pi) / (2*pi) * 255);
imwrite(phase_uint8, fullfile(out_dir, 'phase_8bit.png'));
fprintf('结果已保存到：\n%s\n', out_dir);

info = imfinfo(fullfile(out_dir, 'phase_8bit.png'));
assert(info.Height == Ny && info.Width == Nx && info.BitDepth == 8);
assert(isequal(imread(fullfile(out_dir, 'phase_8bit.png')), phase_uint8));
