%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Trdata,Vldata] = getTrVlData(TrsRGB_dir, TrXYZ_dir, VlsRGB_dir, ...
    VlXYZ_dir, aug, Tr_ImageNum, Vl_ImageNum, imgSize)

Tr_in_images = imageDatastore(TrsRGB_dir, 'ReadFcn', @inRead); %get training imgs

Tr_gt_images = imageDatastore(TrsRGB_dir); %get training imgs GT

Tr_gt_images.ReadFcn = @(filename)gtRead(filename, ...
    TrsRGB_dir, TrXYZ_dir);

if Tr_ImageNum ~=0  && Tr_ImageNum < length(Tr_in_images.Files) %get random images instead of using the entire dataset
    inds = randperm(Tr_ImageNum);
    Tr_in_images.Files = Tr_in_images.Files(inds(1:Tr_ImageNum));
    Tr_gt_images.Files = Tr_gt_images.Files(inds(1:Tr_ImageNum));
end


Vl_in_images = imageDatastore(VlsRGB_dir, 'ReadFcn', @inRead); %get validation imgs

Vl_gt_images = imageDatastore(VlsRGB_dir); %get validation imgs GT

Vl_gt_images.ReadFcn = @(filename)gtRead(filename, ...
    VlsRGB_dir, VlXYZ_dir);

if Vl_ImageNum ~=0  && Vl_ImageNum < length(Vl_in_images.Files) %get random images instead of using the entire dataset
    inds = randperm(Vl_ImageNum);
    Vl_in_images.Files = Vl_in_images.Files(inds(1:Vl_ImageNum));
    Vl_gt_images.Files = Vl_gt_images.Files(inds(1:Vl_ImageNum));
end

if aug == 1
    augmenter = imageDataAugmenter( ...
        'RandScale',[1 1.2], ... %scaling
        'RandXReflection',true, ... %X reflection
        'RandYReflection',true); %Y reflection 
      
        
    Trdata = randomPatchExtractionDatastore(Tr_in_images,Tr_gt_images, ...
        imgSize,'DataAugmentation',augmenter,'PatchesPerImage',...
        1);
elseif aug == 0
    Trdata = randomPatchExtractionDatastore(Tr_in_images,Tr_gt_images, ...
        imgSize,'PatchesPerImage', 1);
end

Vldata = randomPatchExtractionDatastore(Vl_in_images,Vl_gt_images, ...
        imgSize,'PatchesPerImage', 2);