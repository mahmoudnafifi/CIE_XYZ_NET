%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% demo with operators
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Please adjust pp_code/postprocessing.m with your custom functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear
close all

in_dir = fullfile('..','images');

in_ext = '.jpg';

out_dir = fullfile('..','results');

pp_method = 'none|none|denoise+deblur|none|none';
%post-processing methods: denoise, deblur, dehaze, editdetails,
% exposure-fusion, transfer-colors, chrom-adapt, super-res
%The order is: localLayer|sRGB-localLayer|CIE XYZ|sRGB|localLayer

device = 'gpu';

opt = [];  %extra options, if needed

load(fullfile('models','model_sRGB-XYZ-sRGB.mat'));


if exist(out_dir,'dir') == 0
    mkdir(out_dir);
end

filenames = dir(fullfile(in_dir,['*' in_ext]));

filenames = fullfile(in_dir,{filenames(:).name});

for i = 1 : length(filenames)
    filename = filenames{i};
    
    fprintf('processing image %s...\n', filename);
    
    image = im2double(imread(filename));
    
    if length(size(image)) ~= 3
        error('cannot deal with grayscale images');
    end
    
    [~,name,~] = fileparts(filename);
    
    tasks = strsplit(pp_method,'|');
    
    output_XYZ = applyLocalMapping(nets.local_sRGB, image, ...
        'to-xyz', device, tasks{1}, opt);
    
    output_XYZ = applyGlobalMapping(nets.global_sRGB, output_XYZ,device, ...
        tasks{2}, opt);
    
    if strcmpi(tasks{3},'none') == 0
        output_XYZ = postprocessing(output_XYZ,tasks{3},opt);
    end
    
    output_sRGB = applyGlobalMapping(nets.global_XYZ, output_XYZ,device, ...
        tasks{4}, opt);
    
    output_sRGB = applyLocalMapping(nets.local_XYZ, output_sRGB, ...
        'to-srgb', device, tasks{5}, opt);
    
    
    output_sRGB(output_sRGB>1) = 1;
    output_sRGB(output_sRGB<0) = 0;
    
    
    imwrite(output_sRGB,fullfile(out_dir,[name '_result.png']));
    
end


