%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function stop = plotTrainingAccuracy(info,out_dir, check_point_period)

stop = false;
persistent key_


epoch=info.Epoch;
loss=gather(info.TrainingLoss);
iter=info.Iteration;
rmse=info.TrainingRMSE;
accu=info.TrainingAccuracy;
lrate=info.BaseLearnRate;
val_accu=info.ValidationAccuracy;
val_rmse=info.ValidationRMSE;
val_loss = gather(info.ValidationLoss);

M=[epoch,iter,loss,rmse,accu,lrate, val_loss, val_accu, val_rmse];
if info.State == "start"
    key_=char(datetime); key_=strrep(key_,':','-');
    dlmwrite(fullfile(out_dir,sprintf('report_%s.csv',key_)),...
        M,'delimiter',',');
else
    dlmwrite(fullfile(out_dir,sprintf('report_%s.csv',key_)),...
        M,'delimiter',',','-append');
   files=dir(fullfile(out_dir,'*.mat'));
    [~,idx] = sort([files.datenum]);
    files=files(idx);
    if mod(epoch,check_point_period)==0
        if exist(fullfile(out_dir,'backup'),'dir') == 0
            mkdir(fullfile(out_dir,'backup'));
        end
        copyfile(fullfile(out_dir,files(end).name),...
            fullfile(out_dir,'backup',sprintf('epoch_number_%d.mat', ...
            floor(epoch))));
    end
    if length(files)>check_point_period
        for i=1:length(files)-5
            delete(fullfile(out_dir,files(i).name));
        end
    end   
end
end