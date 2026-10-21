function out_dir = make_result_dir(experiment)
% Keep new run products separate from source and historical examples.
if isstring(experiment) && isscalar(experiment)
    experiment = char(experiment);
end
assert(ischar(experiment) && isrow(experiment) && ...
    ~isempty(regexp(experiment, '^[A-Za-z][A-Za-z0-9_]*$', 'once')), ...
    'Experiment name must contain only letters, digits and underscores.');
project_dir = fileparts(fileparts(mfilename('fullpath')));
parent_dir = fullfile(project_dir, 'results', experiment);
if ~isfolder(parent_dir)
    [ok, message] = mkdir(parent_dir);
    assert(ok, '%s', message);
end
out_dir = tempname(parent_dir);
[ok, message] = mkdir(out_dir);
assert(ok, '%s', message);
end
