%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% demo
clc
clear
close all;

in_dir = fullfile('..','images');

in_ext = '.jpg';

device = 'gpu'; %cpu or gpu

load(fullfile('models','model_sRGB-XYZ-sRGB.mat'));

task = 'srgb-2-xyz-2-srgb'; %'srgb-2-xyz-2-srgb', 'srgb-2-xyz', 'xyz-2-srgb'

if strcmpi(task,'srgb-2-xyz-2-srgb') == 0 && ...
        strcmpi(task,'srgb-2-xyz') == 0 && strcmpi(task, 'xyz-2-srgb') == 0
    error(...
        "The task should be: 'srgb-2-xyz-2-srgb', 'srgb-2-xyz', or 'xyz-2-srgb', but the given one is %s", ...
        task);
end

switch task
    case 'srgb-2-xyz-2-srgb'
        out_dir = {fullfile('..','reconstructed_xyz'), ...
            fullfile('..','re-rendered_srgb')};
    case 'srgb-2-xyz'
        out_dir = {fullfile('..','reconstructed_xyz')};
    case 'xyz-2-srgb'
        out_dir = {fullfile('..','re-rendered_srgb')};
end

for o = 1 : length(out_dir)
    if exist(out_dir{o},'dir') == 0
        mkdir(out_dir{o});
    end
end

filenames = dir(fullfile(in_dir, ['*' in_ext]));

filenames = fullfile(in_dir,{filenames(:).name});

for i = 1 : length(filenames)
    
    filename = filenames{i};
    
    fprintf('processing image %s...\n', filename);
    
    [~,name,~] = fileparts(filename);
    
    image = im2double(imread(filename));
    
    switch task
        case 'srgb-2-xyz-2-srgb'
            
            output_XYZ = applyLocalMapping(nets.local_sRGB, image, ...
                'to-xyz', device);
            
            output_XYZ = applyGlobalMapping(nets.global_sRGB, ...
                output_XYZ, device);
            
            output_sRGB = applyGlobalMapping(nets.global_XYZ, ...
                output_XYZ, device);
            
            output_sRGB = applyLocalMapping(nets.local_XYZ, ...
                output_sRGB, 'to-srgb', device);
            
            output_sRGB(output_sRGB>1) = 1;
            
            output_sRGB(output_sRGB<0) = 0;
            
            imwrite(im2uint16(output_XYZ),fullfile(out_dir{1}, ...
                [name '_XYZ_reconstructed.png']));
            
            imwrite(output_sRGB,fullfile(out_dir{2}, ...
                [name '_sRGB_re-rendered.png']));
        
        case 'srgb-2-xyz'
        
            output_XYZ = applyLocalMapping(nets.local_sRGB, image, ...
                'to-xyz', device);            
            
            output_XYZ = applyGlobalMapping(nets.global_sRGB, ...
                output_XYZ, device);
            
            imwrite(im2uint16(output_XYZ),fullfile(out_dir{1}, ...
                [name '_XYZ_reconstructed.png']));

        case 'xyz-2-srgb'
            
            output_sRGB = applyGlobalMapping(nets.global_XYZ, ...
                image, device);
            
            output_sRGB = applyLocalMapping(nets.local_XYZ, ...
                output_sRGB, 'to-srgb', device);
            
            output_sRGB(output_sRGB>1) = 1;
            
            output_sRGB(output_sRGB<0) = 0;
            
            imwrite(output_sRGB,fullfile(out_dir{1}, ...
                [name '_sRGB_re-rendered.png']));  
    end
end
    
