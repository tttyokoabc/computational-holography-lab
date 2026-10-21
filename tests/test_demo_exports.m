function test_demo_exports
% Exercise actual export blocks with full-size canvases and two iterations.
project_dir = fileparts(fileparts(mfilename('fullpath')));
previous = get(groot, 'defaultFigureVisible');
restore = onCleanup(@() set(groot, 'defaultFigureVisible', previous));
set(groot, 'defaultFigureVisible', 'off');
names = {'gs_demo', 'gs_depth_demo', 'gs_two_depth_demo', ...
    'gs_two_depth_weighted'};
for k = 1:numel(names)
    check_export(fullfile(project_dir, 'src', [names{k} '.m']));
end
input_dir = make_result_dir('test_image_input');
input_file = fullfile(input_dir, 'gradient.png');
imwrite(repmat(linspace(0, 1, 16), 16, 1), input_file);
check_export(fullfile(project_dir, 'src', 'gs_image_demo.m'), input_file);
fprintf('PASS: five demos export 1920x1080 8-bit phase PNGs.\n');
end

function check_export(source_path, input_file) %#ok<INUSD>
% input_file is consumed by the image-selection replacement through eval.
code = fileread(source_path);
if nargin == 2
    code = regexprep(code, ...
        '(?s)\[file, folder\] = uigetfile\(.*?''选择目标图片''\);', ...
        '[folder, stem, ext] = fileparts(input_file); file = [stem ext];');
end
code = regexprep(code, '(?m)^clear;\s*$', '');
code = regexprep(code, '(?m)^clc;\s*$', '');
code = regexprep(code, '(?m)^close all;\s*$', '');
code = regexprep(code, 'maxn = \d+;', 'maxn = 2;');
code = regexprep(code, 'record_steps = \[[^\]]+\];', 'record_steps = [1 2];');
eval(code);
if exist('phase_file', 'var')
    png_path = phase_file;
else
    png_path = fullfile(out_dir, 'phase_8bit.png');
end
info = imfinfo(png_path);
assert(info.Width == 1920 && info.Height == 1080 && info.BitDepth == 8);
assert(isequal(imread(png_path), phase_uint8));
assert(startsWith(out_dir, fullfile(fileparts(fileparts(source_path)), 'results')));
close all;
fprintf('PASS: %s export and pixel readback.\n', source_path);
end
