folder = 'train'; % folder deleted already
imds = imageDatastore(folder);

for i = 1:length(imds.Files)
    file = strsplit(imds.Files{i}, '/');
    file = file{end};
    name = strsplit(file, '.');
    name = name{end - 1};
    ker = strsplit(name, '_');
    ker = ker{end};

    if ~exist(strcat('data/', ker), 'dir')
        mkdir(strcat('data/', ker))
    end

    copyfile(strcat(folder, '/', file), strcat('data/', ker, '/', file))
end