%% 1. 初始化
clc;
clear;
close all;

rng(1);                 % 固定随机起点，方便重复比较
Ny = 1080;               % 行数：SLM高度
Nx = 1920;               % 列数：SLM宽度
maxn = 100;              % GS迭代次数

%% 2. 读取目标图片，构造目标振幅

[file, folder] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg;*.bmp', '图像文件'}, ...
    '选择目标图片');

if isequal(file, 0)
    return;                         % 取消选择时停止脚本
end

raw = imread(fullfile(folder, file));

% RGB彩色图转换为灰度；灰度图直接使用
if ndims(raw) == 3
    raw = rgb2gray(raw);
end

% 普通整数图像转换为[0,1]范围的double
gray_image = im2double(raw);

% 等比例缩放，最长边占画布的一半
[h, w] = size(gray_image);
scale = min((Ny/2)/h, (Nx/2)/w);

new_h = max(1, round(h * scale));
new_w = max(1, round(w * scale));

small_image = imresize(gray_image, [new_h new_w]);
small_image = min(max(small_image, 0), 1);

% 将缩放后的图片放在黑色画布中央
target_intensity = zeros(Ny, Nx);

row_start = floor((Ny - new_h)/2) + 1;
col_start = floor((Nx - new_w)/2) + 1;

target_intensity( ...
    row_start:row_start+new_h-1, ...
    col_start:col_start+new_w-1) = small_image;

if sum(target_intensity(:)) == 0
    error('目标图片全黑，请选择包含非零亮度的图片。');
end

% 图片表示期望强度，因此目标振幅取平方根
target_amp = sqrt(target_intensity);

figure('Name', '目标图像');
imagesc(target_intensity, [0 1]);
axis image;
colormap gray;
colorbar;
title('目标强度');

%% 3. 初始化成像面光场
random_phase = 2*pi*rand(Ny, Nx);
im_image = target_amp .* exp(1i*random_phase);

% 记录这些迭代次数对应的重建结果
record_steps = [1 2 5 20 50 100];
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
%% 7. 评价灰度图重建误差

target_norm = target_intensity / sum(target_intensity(:));

K = numel(record_steps);
rmse_value = zeros(K, 1);

for k = 1:K
    I = recon_intensity(:, :, k);
    I_norm = I / sum(I(:));

    difference = I_norm - target_norm;
    rmse_value(k) = sqrt(mean(difference(:).^2));
end

results = table(record_steps(:), rmse_value, ...
    'VariableNames', {'Iteration', 'NormalizedIntensityRMSE'});
disp(results);

figure('Name', '目标图片重建误差');
plot(record_steps, rmse_value, '-o');
xlabel('迭代次数');
ylabel('归一化强度 RMSE');
grid on;
%% 8. 保存本次仿真数据

% 每次新建一个结果文件夹
out_dir = make_result_dir('gs_image_demo');

% 保存完整数值，后面可以重新分析、绘图
save(fullfile(out_dir, 'gs_result.mat'), ...
    'target_intensity', 'target_amp', ...
    'phase_hologram', 'recon_intensity', ...
    'record_steps', 'rmse_value', 'Ny', 'Nx', 'maxn');

% 保存误差表
writetable(results, fullfile(out_dir, 'metrics.csv'));

% 保存目标强度图
imwrite(target_intensity, ...
    fullfile(out_dir, 'target.png'));

% 导出理想线性编码的8位相位图
phase_uint8 = uint8( ...
    (phase_hologram + pi) / (2*pi) * 255);

imwrite(phase_uint8, ...
    fullfile(out_dir, 'phase_8bit.png'));

fprintf('结果已保存到：\n%s\n', out_dir);
info = imfinfo(fullfile(out_dir, 'phase_8bit.png'));
assert(info.Height == Ny && info.Width == Nx && info.BitDepth == 8);
assert(isequal(imread(fullfile(out_dir, 'phase_8bit.png')), phase_uint8));
