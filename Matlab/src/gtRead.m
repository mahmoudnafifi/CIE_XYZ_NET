%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function GTimages = gtRead(fileName, sRGB_dir, XYZ_dir)

sRGB = im2double(imread(fileName));
XYZ = im2double(imread(strrep(strrep(fileName, sRGB_dir, XYZ_dir), ...
    '.JPG','.png')));
if sum(size(XYZ) == size(sRGB)) ~= 3
    XYZ = imresize(XYZ,[size(sRGB,1), size(sRGB,2)]);
end
GTimages = zeros([size(sRGB,1),size(sRGB,2),size(sRGB,3) * 2],'like',sRGB);
GTimages(:,:,1:3) = XYZ;
GTimages(:,:,4:6) = sRGB;



end