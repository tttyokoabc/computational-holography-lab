function test_result_dir
first = make_result_dir('test_result_dir');
second = make_result_dir('test_result_dir');
project_dir = fileparts(fileparts(mfilename('fullpath')));
expected = fullfile(project_dir, 'results', 'test_result_dir');
assert(strcmp(fileparts(first), expected));
assert(isfolder(first) && isfolder(second) && ~strcmp(first, second));
rejected = false;
try
    make_result_dir('../outside');
catch
    rejected = true;
end
assert(rejected, 'Path traversal must be rejected.');
fprintf('PASS: unique run folders inside project results; invalid name rejected.\n');
end
