clear all
close all
clc


project_dir =  './Children/';
for sub_i = 1:27
    sub = sprintf('%3d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    tic
    index_pinky_Tstat_func(sub,ses,project_dir)
    toc
end
% 

                        
for sub_i = 1:27
    sub = sprintf('%3d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    index_pinky_peak_TFS_func(sub,ses,project_dir)
end

for sub_i = 1:27
    sub = sprintf('%3d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    index_pinky_4mm_Tstat_func(sub,ses,project_dir)
end

for sub_i = 1:27
    sub = sprintf('%3d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    index_pinky_4mm_evoked_func(sub,ses,project_dir)
    peak_VE_evoked(sub,ses,project_dir)
end

parpool(4)
for sub_i = 1:27
    sub = sprintf('%3d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    AEC_connectivity_func(sub,ses,project_dir)
end
%
for sub_i = 1:27
    sub = sprintf('%3d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    tic
    data_quality(sub_i,:) = data_quality_func(sub,ses,project_dir)
    toc
end
save([project_dir,'\data_quality_children'],'data_quality')
%%