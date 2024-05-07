%% Make Helmet Info Locs TSV File
clear all
close all
cd R:\OPMMEG\Users\Ryan\App\Helmet_info\ % Helmet info path
fnames = dir('*.mat');
for nnn = 1:size(fnames,1)
    Helmet_info_file = [fnames(nnn).folder '\' fnames(nnn).name];
    Helmet_info_file_save = [Helmet_info_file(1:end-4) '_locs.tsv'];
    load([Helmet_info_file])
    lay = Helmet_info.lay;
    
    Loc_info_file = cell(3*size(Helmet_info.sens_pos,1)+1,8);
    Loc_info_file{1,1} = 'name';
    Loc_info_file{1,2} = 'Px';
    Loc_info_file{1,3} = 'Py';
    Loc_info_file{1,4} = 'Pz';
    Loc_info_file{1,5} = 'Ox';
    Loc_info_file{1,6} = 'Oy';
    Loc_info_file{1,7} = 'Oz';
    Loc_info_file{1,8} = 'Units';
    
    
    for n = 1:size(Helmet_info.sens_pos,1)
        loc_names{n,1} = Helmet_info.sens_labels{n};
    end
    
    for n = 1:size(Helmet_info.sens_pos,1)
        % X axis
        Loc_info_file{3*n-1,1} = [loc_names{n} '(X)'];
        Loc_info_file{3*n-1,2} = Helmet_info.sens_pos(n,1);
        Loc_info_file{3*n-1,3} = Helmet_info.sens_pos(n,2);
        Loc_info_file{3*n-1,4} = Helmet_info.sens_pos(n,3);
        Loc_info_file{3*n-1,5} = Helmet_info.sens_ors_X(n,1);
        Loc_info_file{3*n-1,6} = Helmet_info.sens_ors_X(n,2);
        Loc_info_file{3*n-1,7} = Helmet_info.sens_ors_X(n,3);
        Loc_info_file{3*n-1,8} = 'm';
        
        % Y axis
        Loc_info_file{3*n,1} = [loc_names{n} '(Y)'];
        Loc_info_file{3*n,2} = Helmet_info.sens_pos(n,1);
        Loc_info_file{3*n,3} = Helmet_info.sens_pos(n,2);
        Loc_info_file{3*n,4} = Helmet_info.sens_pos(n,3);
        Loc_info_file{3*n,5} = Helmet_info.sens_ors_Y(n,1);
        Loc_info_file{3*n,6} = Helmet_info.sens_ors_Y(n,2);
        Loc_info_file{3*n,7} = Helmet_info.sens_ors_Y(n,3);
        Loc_info_file{3*n,8} = 'm';
        
        % Z axis
        Loc_info_file{3*n+1,1} = [loc_names{n} '(Z)'];
        Loc_info_file{3*n+1,2} = Helmet_info.sens_pos(n,1);
        Loc_info_file{3*n+1,3} = Helmet_info.sens_pos(n,2);
        Loc_info_file{3*n+1,4} = Helmet_info.sens_pos(n,3);
        Loc_info_file{3*n+1,5} = Helmet_info.sens_ors_Z(n,1);
        Loc_info_file{3*n+1,6} = Helmet_info.sens_ors_Z(n,2);
        Loc_info_file{3*n+1,7} = Helmet_info.sens_ors_Z(n,3);
        Loc_info_file{3*n+1,8} = 'm';
        
    end
    
    % Save to file
    fid_locs = fopen(Helmet_info_file_save, 'wt' );
    i = 1;
    fprintf(fid_locs,'%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n',Loc_info_file{i,1},Loc_info_file{i,2},...
        Loc_info_file{i,3},Loc_info_file{i,4},Loc_info_file{i,5},...
        Loc_info_file{i,6},Loc_info_file{i,7},Loc_info_file{i,8});
    for i = 2:size(Loc_info_file,1)
        clc
        disp(['Writing line: ' num2str(i)])
        fprintf(fid_locs,'%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n',Loc_info_file{i,1},Loc_info_file{i,2},...
            Loc_info_file{i,3},Loc_info_file{i,4},Loc_info_file{i,5},...
            Loc_info_file{i,6},Loc_info_file{i,7},Loc_info_file{i,8});
        %                 sprintf('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n',Loc_info_file{i,1},Loc_info_file{i,2},...
        %                     Loc_info_file{i,3},Loc_info_file{i,4},Loc_info_file{i,5},...
        %                     Loc_info_file{i,6},Loc_info_file{i,7},Loc_info_file{i,8})
    end
    fclose(fid_locs);
end

%% Make Helmet Info TSV File
clear all
close all
cd R:\OPMMEG\Users\Ryan\App\Helmet_info\ % Helmet info path
fnames = dir('*.mat');
for nnn = 1:size(fnames,1)
    Helmet_info_file = [fnames(nnn).folder '\' fnames(nnn).name];
    Helmet_info_file_save = [Helmet_info_file(1:end-4) '_info.tsv'];
    load([Helmet_info_file])
    lay = Helmet_info.lay;
    
    Loc_info_file = cell(3*size(Helmet_info.sens_pos,1)+1,8);
    Loc_info_file{1,1} = 'Name';
    Loc_info_file{1,2} = 'Px';
    Loc_info_file{1,3} = 'Py';
    Loc_info_file{1,4} = 'Pz';
    Loc_info_file{1,5} = 'Ox';
    Loc_info_file{1,6} = 'Oy';
    Loc_info_file{1,7} = 'Oz';
    Loc_info_file{1,8} = 'Layx';
    Loc_info_file{1,9} = 'Layy';
       
    for n = 1:size(Helmet_info.sens_pos,1)
        loc_names{n,1} = Helmet_info.sens_labels{n};
    end
    
    for n = 1:size(Helmet_info.sens_pos,1)
        % X axis
        Loc_info_file{3*n-1,1} = [loc_names{n} ' X'];
        Loc_info_file{3*n-1,2} = Helmet_info.sens_pos(n,1);
        Loc_info_file{3*n-1,3} = Helmet_info.sens_pos(n,2);
        Loc_info_file{3*n-1,4} = Helmet_info.sens_pos(n,3);
        Loc_info_file{3*n-1,5} = Helmet_info.sens_ors_X(n,1);
        Loc_info_file{3*n-1,6} = Helmet_info.sens_ors_X(n,2);
        Loc_info_file{3*n-1,7} = Helmet_info.sens_ors_X(n,3);
        Loc_info_file{3*n-1,8} = Helmet_info.lay.pos(n,1);
        Loc_info_file{3*n-1,9} = Helmet_info.lay.pos(n,2);
        
        % Y axis
        Loc_info_file{3*n,1} = [loc_names{n} ' Y'];
        Loc_info_file{3*n,2} = Helmet_info.sens_pos(n,1);
        Loc_info_file{3*n,3} = Helmet_info.sens_pos(n,2);
        Loc_info_file{3*n,4} = Helmet_info.sens_pos(n,3);
        Loc_info_file{3*n,5} = Helmet_info.sens_ors_Y(n,1);
        Loc_info_file{3*n,6} = Helmet_info.sens_ors_Y(n,2);
        Loc_info_file{3*n,7} = Helmet_info.sens_ors_Y(n,3);
        Loc_info_file{3*n,8} = Helmet_info.lay.pos(n,1);
        Loc_info_file{3*n,9} = Helmet_info.lay.pos(n,2);
        
        % Z axis
        Loc_info_file{3*n+1,1} = [loc_names{n} ' Z'];
        Loc_info_file{3*n+1,2} = Helmet_info.sens_pos(n,1);
        Loc_info_file{3*n+1,3} = Helmet_info.sens_pos(n,2);
        Loc_info_file{3*n+1,4} = Helmet_info.sens_pos(n,3);
        Loc_info_file{3*n+1,5} = Helmet_info.sens_ors_Z(n,1);
        Loc_info_file{3*n+1,6} = Helmet_info.sens_ors_Z(n,2);
        Loc_info_file{3*n+1,7} = Helmet_info.sens_ors_Z(n,3);
        Loc_info_file{3*n+1,8} = Helmet_info.lay.pos(n,1);
        Loc_info_file{3*n+1,9} = Helmet_info.lay.pos(n,2);
        
    end
    
    % Save to file
    fid_locs = fopen(Helmet_info_file_save, 'wt' );
    i = 1;
    fprintf(fid_locs,'%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n',Loc_info_file{i,1},Loc_info_file{i,2},...
        Loc_info_file{i,3},Loc_info_file{i,4},Loc_info_file{i,5},...
        Loc_info_file{i,6},Loc_info_file{i,7},Loc_info_file{i,8},Loc_info_file{i,8});
    for i = 2:size(Loc_info_file,1)
        clc
        disp(['Writing line: ' num2str(i)])
        fprintf(fid_locs,'%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n',Loc_info_file{i,1},Loc_info_file{i,2},...
            Loc_info_file{i,3},Loc_info_file{i,4},Loc_info_file{i,5},...
            Loc_info_file{i,6},Loc_info_file{i,7},Loc_info_file{i,8},Loc_info_file{i,9});
        %                 sprintf('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n',Loc_info_file{i,1},Loc_info_file{i,2},...
        %                     Loc_info_file{i,3},Loc_info_file{i,4},Loc_info_file{i,5},...
        %                     Loc_info_file{i,6},Loc_info_file{i,7},Loc_info_file{i,8})
    end
    fclose(fid_locs);
end