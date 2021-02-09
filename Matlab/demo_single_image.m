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

filename = fullfile('..','images','a0280-IMG_0854.JPG');

device = 'gpu'; %cpu or gpu

load(fullfile('models','model_sRGB-XYZ-sRGB.mat'));

task = 'srgb-2-xyz-2-srgb'; %'srgb-2-xyz-2-srgb', 'srgb-2-xyz', 'xyz-2-srgb'

show = 1;

save_output = 1;

gt_image_ext = '.png';

gt_dir = '..\XYZ_testing';

if strcmpi(task,'srgb-2-xyz-2-srgb') == 0 && ...
        strcmpi(task,'srgb-2-xyz') == 0 && strcmpi(task, 'xyz-2-srgb') == 0
    error(...
        "The task should be: 'srgb-2-xyz-2-srgb', 'srgb-2-xyz', or 'xyz-2-srgb', but the given one is %s", ...
        task);
end

if save_output == 1
    
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
    
end


fprintf('processing image %s...\n', filename);

[~,name,~] = fileparts(filename);

gt_name = [name gt_image_ext];

if exist(fullfile(gt_dir,gt_name),'file') == 0
    disp('Warning: cannot find ground truth image');
    noGT = 1;
else
    
    GT = im2double(imread(fullfile(gt_dir,gt_name)));
    noGT = 0;
end
image = im2double(imread(filename));

switch task
    case 'srgb-2-xyz-2-srgb'
        
        output_XYZ = applyLocalMapping(nets.local_sRGB, image, ...
            'to-xyz', device);
        
        output_XYZ = applyGlobalMapping(nets.global_sRGB, ...
            output_XYZ,device);
        
        output_sRGB = applyGlobalMapping(nets.global_XYZ, ...
            output_XYZ,device);
        
        output_sRGB = applyLocalMapping(nets.local_XYZ, output_sRGB, ...
            'to-srgb', device);
        
        
        output_sRGB(output_sRGB>1) = 1;
        output_sRGB(output_sRGB<0) = 0;
        
        if save_output == 1
            imwrite(im2uint16(output_XYZ),fullfile(out_dir{1},...
                [name '_XYZ_reconstructed.png']));
            imwrite(output_sRGB,fullfile(out_dir{2}, ...
                [name '_sRGB_re-rendered.png']));
        end
        
        if show == 1
            if noGT == 1
                subplot(1,4,1);imshow(image); 
                title('input');
                subplot(1,4,2);imshow(rgb2xyz(image));
                title('standard');
                subplot(1,4,3);imshow(output_XYZ); 
                title('ours');
                subplot(1,4,4);imshow(output_sRGB);
                title('re-rendered sRGB');
            else
                subplot(1,5,1);imshow(image);
                title('input');
                subplot(1,5,2);imshow(rgb2xyz(image));
                title('standard');
                subplot(1,5,3);imshow(output_XYZ);
                title('ours');
                subplot(1,5,4);imshow(GT);
                title('GT');
                subplot(1,5,5);imshow(output_sRGB);
                title('re-rendered sRGB');
                
            end
            linkaxes
        end
        
    case 'srgb-2-xyz'
        
        output_XYZ = applyLocalMapping(nets.local_sRGB, image, ...
            'to-xyz', device);
        
        output_XYZ = applyGlobalMapping(nets.global_sRGB, ...
            output_XYZ, device);
        
        if save_output == 1
            imwrite(im2uint16(output_XYZ),fullfile(out_dir{1}, ...
                [name '_XYZ_reconstructed.png']));
        end
        
        if show == 1
            if noGT == 1
                subplot(1,3,1);imshow(image);  title('input');
                subplot(1,3,2);imshow(rgb2xyz(image)); title('standard');
                subplot(1,3,3);imshow(output_XYZ); title('ours');
            else
                subplot(1,4,1);imshow(image);  title('input');
                subplot(1,4,2);imshow(rgb2xyz(image)); title('standard');
                subplot(1,4,3);imshow(output_XYZ); title('ours');
                subplot(1,4,4);imshow(GT); title('GT');
            end
            linkaxes
        end
    case 'xyz-2-srgb'
        output_sRGB = applyGlobalMapping(nets.global_XYZ, ...
            image, device);
        
        output_sRGB = applyLocalMapping(nets.local_XYZ, output_sRGB, ...
            'to-srgb', device);
        
        output_sRGB(output_sRGB>1) = 1;
        output_sRGB(output_sRGB<0) = 0;
        
        if save_output == 1
            imwrite(output_sRGB,fullfile(out_dir{1}, ...
                [name '_sRGB_re-rendered.png']));
        end
        if show == 1
            if noGT == 1
                subplot(1,3,1);imshow(image);  title('input');
                subplot(1,3,2);imshow(xyz2rgb(image)); title('standard');
                subplot(1,3,3);imshow(output_sRGB); title('ours');
            else
                subplot(1,4,1);imshow(image);  title('input');
                subplot(1,4,2);imshow(xyz2rgb(image)); title('standard');
                subplot(1,4,3);imshow(output_sRGB); title('ours');
                subplot(1,4,4);imshow(GT); title('GT');
            end
        end
        linkaxes
end

