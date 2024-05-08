clear all
close all
clc
if badtrls
    project_dir =  '.\Adults_badtrls\';
else
    project_dir =  '.\Adults\';

end
subs = [1:26];
for sub_i = subs
    sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    tic
    index_pinky_Tstat_func(sub,ses,project_dir)
    toc
end
%
                        
for sub_i = subs
    sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    index_pinky_peak_TFS_func(sub,ses,project_dir)
end
% %
%%

for sub_i = subs
    sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    index_pinky_4mm_Tstat_func(sub,ses,project_dir)
end
%
for sub_i = subs
    sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    index_pinky_4mm_evoked_func(sub,ses,project_dir)
    peak_VE_evoked(sub,ses,project_dir)
end

%
parpool(4)
for sub_i = subs
    sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    AEC_connectivity_func(sub,ses,project_dir)
end
%
for sub_i = subs
    sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    tic
    data_quality(sub_i,:) = data_quality_func(sub,ses,project_dir);
    toc
end
save([project_dir,'\data_quality_adults'],'data_quality')
