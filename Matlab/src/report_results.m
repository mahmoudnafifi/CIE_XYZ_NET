%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function results = report_results(base_dir,out_full_name)

files = dir(fullfile(base_dir,'*.mat'));


PSNR = zeros(length(files),1);
MAE = zeros(size(PSNR));
names = {files(:).name};
for i = 1 : length(files)
    load(fullfile(base_dir,files(i).name));
    PSNR(i) = result.PSNR;
end

mean_PSNR = mean(PSNR);
q1_PSNR = q1(PSNR); 
q2_PSNR = q2(PSNR); 
q3_PSNR = q3(PSNR); 

results.PSNR = PSNR; results.names = names;


results.mean_PSNR = mean_PSNR; 

results.q1_PSNR = q1_PSNR; 

results.q2_PSNR = q2_PSNR; 

results.q3_PSNR = q3_PSNR; 

save(out_full_name,'results','-v7.3');

metrics = {'PSNR'};
for e = 1 : length(metrics)
    fprintf('%s:\n',metrics{e});
    mean_ = eval(sprintf('mean(results.%s);',metrics{e}));
    q1_ = eval(sprintf('q1(results.%s);',metrics{e}));
    q2_ = eval(sprintf('q2(results.%s);',metrics{e}));
    q3_ = eval(sprintf('q3(results.%s);',metrics{e}));
    fprintf('mean = %.2f, Q1 = %.2f, Q2 = %.2f, Q3 = %.2f\n',...
        mean_,q1_,q2_,q3_);
end

end

function a = q1 (error)
a = quantile(error,0.25);
end

function a = q2 (error)
a = quantile(error,0.5);
end

function a = q3 (error)
a = quantile(error,0.75);
end