singletrack_ana()
function singletrack_ana()
    addpath 'C:\Program Files (x86)\mtex-5.9.0\mtex-5.9.0'
    startup_mtex
    %% Specify Crystal and Specimen Symmetries
    
    % crystal symmetry
    CS = {... 
      'notIndexed',...
      crystalSymmetry('m-3m', [3.6 3.6 3.6], 'mineral', 'Nickel', 'color', [0.53 0.81 0.98])};
    
    % plotting convention
    setMTEXpref('xAxisDirection','west');
    setMTEXpref('zAxisDirection','outOfPlane');
    
    %% Specify File Names
    
    % path to files
    pname = 'D:\JianTANG\CA\HastelloyX_validation\SingleTrack-EBSD-data\SingleTrack-EBSD-data\'; %'D:\JianTANG\Work\WorkRecord\Experiment\SingleTrack-HX\microstructure\AR-CLD-bnd-master\AR-CLD-bnd-master';
    namei = {'E1P1-0-1.ctf'; 'E1P1-0-3.ctf'; 'E1P2-0-2.ctf'; 'E1P2-0-5.ctf'; 'E1P3-0-1.ctf'; 'E1P3-0-3.ctf'; 
        'E2P1-0-2.ctf'; 'E2P1-0-4.ctf'; 'E2P2-0-1.ctf'; 'E2P2-0-5.ctf'; 'E2P3-0-4.ctf'; 'E2P3-0-4.ctf';
        'E3P1-0-1.ctf'; 'E3P1-0-3.ctf'; 'E3P2-0-2.ctf'; 'E3P2-0-5.ctf'; 'E3P3-0-2.ctf'; 'E3P3-0-5.ctf'};
    
    stats_rec=[];
    base_diff=[];
    cld_file = [pname '\cld_rec.xlsx'];
    for i = 1:9
        stats_pc=[];
        for j = 1:2
            % which files to be imported
            fname = [pname+string(namei((i-1)*2+j))];
            stats_tmp = ebsd_ana(fname,CS,'mergeTwin','saveIPF');
            stats_pc=[stats_pc;stats_tmp];
            writematrix(stats_tmp.CLD',cld_file,'Sheet',string(namei((i-1)*2+j)));
        end
        [d_tmp] = com_two_dis(stats_pc(1), stats_pc(2));
        base_diff = [base_diff; d_tmp];
        stats_pc_tmp = stats_pc(1);
        stats_pc_tmp.CLD = (stats_pc(1).CLD+stats_pc(2).CLD)/2.0;
        stats_rec = [stats_rec;stats_pc_tmp];

        % plot the CLD
        pltCLD= [stats_pc_tmp.CLD;stats_pc_tmp.CLD(1,:)];
        plot_CLD(stats_pc_tmp.hst_x,pltCLD,'in_ang',(0:10:180)*pi/180,'clim',[0,0.15])
        saveas(gcf,string(extractBefore(fname,'.'))+'_CLD.png')

        bar(stats_pc_tmp.hst_x,mean(stats_pc_tmp.CLD,1))
        ylim([0,0.3])
        xlabel("Chord length (um)")
        ylabel("Frequency")
        saveas(gcf,string(extractBefore(fname,'.'))+'_CLDm.png')
        
%         for k=1:3:size(stats_pc_tmp.CLD,1)-1
%             figure;
%             bar(stats_pc_tmp.hst_x,stats_pc_tmp.CLD(k,:)-mean(stats_pc_tmp.CLD,1))
%             ylim([0,0.15])
%             xlabel("Chord length (um)")
%             ylabel("Frequency")
%             saveas(gcf,string(extractBefore(fname,'.'))+'_CLD_ang'+string(k)+'.png')
%         end
    end
    writematrix(base_diff,cld_file,'Sheet','base_diff_pc');

    filename = [pname 'pc_diff.xlsx'];
    pc_diff_rec=[];
    pc_diff_ave_rec = [];
    for i =1:9
        pc_diff=[];
        pc1_name = string(namei(i*2));
        pc1 = stats_rec(i);
        for j=1:9
            if i == j
                continue;
            end
            pc2_name = string(namei(j*2));
            pc2 = stats_rec(j);
            [d_tmp] = com_two_dis(pc1,pc2);
            pc_diff=[pc_diff;pc2_name d_tmp];
        end
        pc_diff_rec = [pc_diff_rec pc_diff(:,end)];
        pc_diff_ave_rec = [pc_diff_ave_rec pc_diff(:,end-1)];
        writematrix(pc_diff,filename,'Sheet',pc1_name);
    end
    pc_diff_rec =pc_diff_rec';
    pc_diff_rec_tmp=zeros(9,9);
    for i =1:9
        for j=1:9
            if j >i
                pc_diff_rec_tmp(i,j) = pc_diff_rec(i,j-1);
            elseif j<i
                pc_diff_rec_tmp(i,j) = pc_diff_rec(i,j);
            else
                pc_diff_rec_tmp(i,j) = 0;
            end
        end
    end
    writematrix(pc_diff_rec_tmp,filename,'Sheet','pc_diff_normdist');

    pc_diff_ave_rec =pc_diff_ave_rec';
    pc_diff_rec_tmp=zeros(9,9);
    for i =1:9
        for j=1:9
            if j >i
                pc_diff_rec_tmp(i,j) = pc_diff_ave_rec(i,j-1);
            elseif j<i
                pc_diff_rec_tmp(i,j) = pc_diff_ave_rec(i,j);
            else
                pc_diff_rec_tmp(i,j) = 0;
            end
        end
    end
    writematrix(pc_diff_rec_tmp,filename,'Sheet','pc_diff_mean_d');
end

function [dtmp] = sim_exp_com(fname_exp,fname_sim)
stats_exp = ebsd_ana(fname_exp,CS,'mergeTwin','saveIPF');
pltCLD= [stats_exp.CLD;stats_exp.CLD(1,:)];
plot_CLD(stats_exp.hst_x,pltCLD,'in_ang',(0:10:180)*pi/180,'clim',[0,0.15])
saveas(gcf,string(extractBefore(fname_exp,'.'))+'_CLD.png')

stats_sim = ebsd_ana(fname_sim,CS,'mergeTwin','saveIPF','cho_max',100*1e-6);
stats_sim.hst_x = stats_sim.hst_x * 1e6;
pltCLD= [stats_sim.CLD;stats_sim.CLD(1,:)];
plot_CLD(stats_sim.hst_x,pltCLD,'in_ang',(0:10:180)*pi/180,'clim',[0,0.15]);
saveas(gcf,'Sim_'+string(extractBefore(fname_sim,'.'))+'_CLD.png');

bar(stats_exp.hst_x,mean(stats_exp.CLD,1))
hold on
bar(stats_sim.hst_x,mean(stats_sim.CLD,1))
alpha(0.8)
ylim([0,0.3])
xlabel("Chord length (um)")
ylabel("Frequency")
legend("Exp","Sim")
saveas(gcf,'Sim_'+string(extractBefore(fname_sim,'.'))+'_CLDm.png')

[d_tmp] = com_two_dis(stats_exp, stats_sim)

writematrix(stats_exp.CLD','Sim_'+string(extractBefore(fname_sim,'.'))+'.xlsx','Sheet',fname_exp);
writematrix(stats_sim.CLD','Sim_'+string(extractBefore(fname_sim,'.'))+'.xlsx','Sheet','Sim_'+fname_exp);
writematrix(d_tmp,'Sim_'+string(extractBefore(fname_sim,'.'))+'.xlsx','Sheet','Sim_Exp_WD');
end

function [I_rec] = extractExpSub(fname,CS)
    %% Import the Data
    % create an EBSD variable containing the data
    ebsd = EBSD.load(fname,CS,'interface', extractAfter(fname,'.'),...
      'convertSpatial2EulerReferenceFrame', 'setting 3');
    
    
    [grains,ebsd.grainId] = calcGrains(ebsd('Nickel'),'angle',3*degree);
    grains = smooth(grains,3);
    
    figure
    ipfKey = ipfColorKey(ebsd('Nickel'));
    ipfKey.inversePoleFigureDirection = vector3d.Y;   % colors follow orientations in y direction
    colors = ipfKey.orientation2color(ebsd('Nickel').orientations);
    plot(ebsd('Nickel'),colors)
    
    cen_x = input("what is the x coordinate of meltpool centre: ");
    cen_y = input("what is the y coordinate of meltpool centre: ");

    rec=[ebsd.x-cen_x -ebsd.y-cen_y ebsd.orientations.phi1 ebsd.orientations.Phi ebsd.orientations.phi2 ebsd.grainId];
%     rec(rec(:,2)>0,:)=[];
%     rec(mod(abs(rec(:,1)),1.5)~=0,:)= [];
%     rec(mod(abs(rec(:,2)),1.5)~=0,:)= [];
%     rec(:,1:2)=rec(:,1:2)/1.5;
    rec(rec(:,2)>0,:)=[];
    rec(:,1:2)=rec(:,1:2)/0.5;
    rec(mod(abs(rec(:,1)),3)~=0,:)= [];
    rec(mod(abs(rec(:,2)),3)~=0,:)= [];
    rec(:,1:2)=rec(:,1:2)/3;

    I=zeros(140,140,6);
    for i =1:size(rec,1)
        %if round(rec(i,1)+70)>0 && round(rec(i,1)+70)<141 && round(rec(i,2)+140)>0 && round(rec(i,2)+140)<141
            I(round(rec(i,1)+70),round(rec(i,2)+140),:)=rec(i,:);
        %end
    end
    I_rec=I(:,:,3:6);
    unigrain=unique(I_rec(:,:,4));
    for i =1:140
        for j=1:140
            I_rec(i,j,4)=find(unigrain==I_rec(i,j,4))-1;
        end
    end
    
end

function [d_tmp] = com_two_dis(stats1, stats2)
    d_tmp =[];
    % compare two CLDs at different angle
    for i =1:size(stats1.CLD,1)
        f1=stats1.hst_x;f2=stats2.hst_x;w1=stats1.CLD(i,:);w2=stats2.CLD(i,:);
        [d, flow] = emd_ca(f1',f2',w1,w2);
        d_tmp = [d_tmp d];
    end
    d_tmp = [d_tmp mean(d_tmp)];
    % first average the distribution and then check the difference
    f1=stats1.hst_x;f2=stats2.hst_x;w1=mean(stats1.CLD,1);w2=mean(stats2.CLD,1);
    [d, flow] = emd_ca(f1',f2',w1,w2);
    d_tmp = [d_tmp d];
end
function [d, flow] = emd_ca(f1,f2,w1,w2)
% subrountine for calculate the wasserstein distance between two
% distribution
% f1, f2: the bin of each bar; size=n*1
% w1, w2: the normalized weight for each bin; size=1*n

% calculate the ground distance matrix (cost matrix)
C = gdm(f1, f2, @gdf);

% calculate the wasserstein distance
[d,flow]=emd_mex(w1,w2,C);
end

function [stats] = ebsd_ana(fname,CS,varargin)
    % list of directions in deg
    angs = 0:10:179;
    % chord max in histogram
    cho_max = 120;%1.2*max(grains.diameter);
    % bin number in histogram
    cho_nbins = 20;
    % merge the twins or not
    mergeT = 0;
    % save the IPF plot or not
    saveipf = 0;
    
    % filter grains with size smaller than 4 pixel
    g_fil =4;
    min_cl = 3;
    cut_d = 0;

    %% digest input 
    if nargin > 1
        for ii=1:length(varargin)
            if strcmp(varargin{ii},'angle') == 1
                angs = varargin{ii+1};
                disp('Use the imported angle range for rotations...')
            end
            if strcmp(varargin{ii},'mergeTwin') == 1
                mergeT = 1;
                disp('The twins will be merged')
            end
            if strcmp(varargin{ii},'cho_max') == 1
                cho_max = varargin{ii+1};
                disp('Use the imported maximum chord for bar plot...')
            end
            if strcmp(varargin{ii},'cho_nbins') == 1
                cho_nbins = varargin{ii+1};
                disp('Use the imported bin number for bar plot...')
            end
            if strcmp(varargin{ii},'grainfilter') == 1
                g_fil = varargin{ii+1};
            end
            if strcmp(varargin{ii},'saveIPF') == 1
                saveipf=1;
            end
            if strcmp(varargin{ii},'min_cl') == 1
                min_cl=varargin{ii+1};
            end
            if strcmp(varargin{ii},'cut') == 1
                cut_d = 1;
                x_l=varargin{ii+1};
                x_r=varargin{ii+2};
                y_l=varargin{ii+3};
                y_r=varargin{ii+4};
            end
        end
    end

    %% Import the Data
    % create an EBSD variable containing the data
    ebsd = EBSD.load(fname,CS,'interface', extractAfter(fname,'.'),...
      'convertSpatial2EulerReferenceFrame', 'setting 3');
    
    if cut_d ~=0
        region=[x_l y_l x_r y_r];
        condition = inpolygon(ebsd,region);
        ebsd = ebsd(condition);
    end
    [grains,ebsd.grainId] = calcGrains(ebsd('indexed'),'angle',10*degree);
    %grains = smooth(grains,3);
    %large_grains = grains(grains.grainSize > 4);
    %ebsd = ebsd(large_grains);
    ebsd(grains(grains.grainSize<g_fil)) = [];
    ebsd('notIndexed')=[];
    
    F = halfQuadraticFilter;
    F.alpha = 0.25;
    
    % interpolate the missing data
    ebsd = smooth(ebsd,F,'fill',grains);%


    % segment grains in MTEX, using, say 10 deg threshold
    grains = calcGrains(ebsd('indexed'),'angle',10*degree);
    grains = smooth(grains,3);
    

    % define twinning misorientation
    CS = grains.CS;
    twinning = orientation('map', Miller(1, 1, 1, CS), Miller(1, 1, 1, CS), Miller(1, -1, 0, CS), Miller(-1, 1, 0, CS));%orientation.map(Miller({2 3 1},CS),Miller({0 1 0},CS),Miller({2 -1 -1},CS),Miller({2 0 -3},CS));
    
    % extract all Magnesium Magnesium grain boundaries
    gB = grains.boundary('Nickel','Nickel');
    
    % and check which of them are twinning boundaries with threshold 5 degree
    isTwinning = angle(gB.misorientation,twinning) < 5*degree;
    twinBoundary = gB(isTwinning);
    [mergedGrains,parentId] = merge(grains,twinBoundary);
    
    if mergeT==0
        mergedGrains=grains;
    end
        
    figure
    ipfKey = ipfColorKey(ebsd('Nickel'));
    ipfKey.inversePoleFigureDirection = vector3d.Y;   % colors follow orientations in y direction
    colors = ipfKey.orientation2color(ebsd('Nickel').orientations);
    plot(ebsd('Nickel'),colors)
    hold on
    plot(mergedGrains.boundary,'linewidth',1.5)
    hold off 
    if saveipf == 1
        saveas(gcf,string(extractBefore(fname,'.'))+'_IPF.png')
        disp('Saving the IPF map...')
    end

    % get grain boundary segments
    gbs = gb2gbs(mergedGrains.boundary);
    
    %% set calculation settings
    % spacing between test lines
    dxy = max(ebsd.unitCell) - min(ebsd.unitCell);
    spacing = min(dxy);
   
    %% main calculation
    % calculate AR-CLD
    stats = calc_CLD(gbs,spacing,angs,'cho_max',cho_max,'silent','cho_nbins',cho_nbins,'min_cl',min_cl);%,'include'

    %% pole figure
    % compute optimal halfwidth from the meanorientations of grains
    psi = calcKernel(grains('Nickel').meanOrientation);
    
    % compute the ODF with the kernel psi
    figure;
    odf = calcDensity(ebsd('Nickel').orientations,'kernel',psi);
    h = [Miller(1,0,0,odf.CS),Miller(1,1,0,odf.CS),Miller(1,1,1,odf.CS)];
    plotPDF(odf,h,'antipodal','silent')
end
function [] = ave_CLD()
namei = {'E1P1-0-1.ctf'; 'E1P1-0-3.ctf'; 'E1P2-0-2.ctf'; 'E1P2-0-5.ctf';  
        'E2P1-0-2.ctf'; 'E2P1-0-4.ctf'; 'E2P2-0-1.ctf'; 'E2P2-0-5.ctf'; 'E2P3-0-4.ctf'; 'E2P3-0-4.ctf';};
fname = pname + "E1P2-0-5.ctf";
stats_tmp = ebsd_ana(fname,CS,'saveIPF','cho_max',120,'grainfilter',12,'cho_nbins',24,'cut',0,170,0,-140);

fname=pname+"E2P3-0-2.ctf"
stats_tmp_pc52 = ebsd_ana(fname,CS,'saveIPF','grainfilter',12,'cho_nbins',24);
ebsd.x=ebsd.x/0.93*0.5;ebsd.y=ebsd.y/0.93*0.5;
ebsd.unitCell = calcUnitCell([ebsd.prop.x(:),ebsd.prop.y(:)]);
region=[0 0 170 -140];%21 -26.5 140 -130]
condition = inpolygon(ebsd,region);
ebsd = ebsd(condition);

fname=pname+"E2P3-0-4.ctf"
stats_tmp = ebsd_ana(fname,CS,'saveIPF');
region=[0 0 170 -130];%21 -26.5 140 -130]
condition = inpolygon(ebsd,region);
ebsd = ebsd(condition);

fname=pname+"E1P1-0-1.ctf"
stats_tmp_pc11 = ebsd_ana(fname,CS,'saveIPF','grainfilter',12,'cho_nbins',24);
region=[min(ebsd.x) max(ebsd.y) 170 -140]
condition = inpolygon(ebsd,region);
ebsd = ebsd(condition);
fname=pname+"E1P1-0-3.ctf"
stats_tmp_pc12 = ebsd_ana(fname,CS,'saveIPF','cho_nbins',24,'grainfilter',12);
region=[min(ebsd.x) max(ebsd.y) 140 -130]
condition = inpolygon(ebsd,region);
ebsd = ebsd(condition);
fname=pname+"E1P2-0-2.ctf"
stats_tmp_pc21 = ebsd_ana(fname,CS,'saveIPF','grainfilter',12,'cho_nbins',24);
region=[0 0 170 -160];%21 -26.5 140 -130]
condition = inpolygon(ebsd,region);
ebsd = ebsd(condition);
fname=pname+"E1P2-0-5.ctf"
stats_tmp_pc22 = ebsd_ana(fname,CS,'saveIPF','grainfilter',12,'cho_nbins',24);
region=[0 0 170 -140];%21 -26.5 140 -130]
condition = inpolygon(ebsd,region);
ebsd = ebsd(condition);
fname=pname+"E2P1-0-2.ctf"
stats_tmp_pc31 = ebsd_ana(fname,CS,'saveIPF','grainfilter',12,'cho_nbins',24);
region=[0 0 170 -140];%21 -26.5 140 -130]
condition = inpolygon(ebsd,region);
ebsd = ebsd(condition);
fname=pname+"E2P1-0-4.ctf"
stats_tmp_pc32 = ebsd_ana(fname,CS,'saveIPF','grainfilter',12,'cho_nbins',24);
region=[0 0 160 -130];%21 -26.5 140 -130]
condition = inpolygon(ebsd,region);
ebsd = ebsd(condition);
fname=pname+"E2P2-0-1.ctf"
stats_tmp_pc41 = ebsd_ana(fname,CS,'saveIPF','grainfilter',12,'cho_nbins',24);
region=[0 0 180 -150];%21 -26.5 140 -130]
condition = inpolygon(ebsd,region);
ebsd = ebsd(condition);
fname=pname+"E2P2-0-5.ctf"
stats_tmp_pc42 = ebsd_ana(fname,CS,'saveIPF','grainfilter',12,'cho_nbins',24);
region=[0 0 180 -150];%21 -26.5 140 -130]
condition = inpolygon(ebsd,region);
ebsd = ebsd(condition);
end

function [grains] = grain_ana(fname,CS,varargin)
    % list of directions in deg
    angs = 0:10:179;
    % chord max in histogram
    cho_max = 120;%1.2*max(grains.diameter);
    % bin number in histogram
    cho_nbins = 20;
    % merge the twins or not
    mergeT = 0;
    % save the IPF plot or not
    saveipf = 0;
    
    % filter grains with size smaller than 4 pixel
    g_fil =4;
    min_cl = 3;
    cut_d = 0;

    %% digest input 
    if nargin > 1
        for ii=1:length(varargin)
            if strcmp(varargin{ii},'angle') == 1
                angs = varargin{ii+1};
                disp('Use the imported angle range for rotations...')
            end
            if strcmp(varargin{ii},'mergeTwin') == 1
                mergeT = 1;
                disp('The twins will be merged')
            end
            if strcmp(varargin{ii},'grainfilter') == 1
                g_fil = varargin{ii+1};
            end
            if strcmp(varargin{ii},'saveIPF') == 1
                saveipf=1;
            end
            if strcmp(varargin{ii},'min_cl') == 1
                min_cl=varargin{ii+1};
            end
            if strcmp(varargin{ii},'cut') == 1
                cut_d = 1;
                x_l=varargin{ii+1};
                x_r=varargin{ii+2};
                y_l=varargin{ii+3};
                y_r=varargin{ii+4};
            end
        end
    end

    %% Import the Data
    % create an EBSD variable containing the data
    ebsd = EBSD.load(fname,CS,'interface', extractAfter(fname,'.'),...
      'convertSpatial2EulerReferenceFrame', 'setting 3');
    
    if cut_d ~=0
        region=[x_l y_l x_r y_r];
        condition = inpolygon(ebsd,region);
        ebsd = ebsd(condition);
    end
    [grains,ebsd.grainId] = calcGrains(ebsd('indexed'),'angle',10*degree);
    %grains = smooth(grains,3);
    %large_grains = grains(grains.grainSize > 4);
    %ebsd = ebsd(large_grains);
    ebsd(grains(grains.grainSize<g_fil)) = [];
    ebsd('notIndexed')=[];
    
    F = halfQuadraticFilter;
    F.alpha = 0.25;
    
    % interpolate the missing data
    ebsd = smooth(ebsd,F,'fill',grains);%


    % segment grains in MTEX, using, say 10 deg threshold
    grains = calcGrains(ebsd('indexed'),'angle',10*degree);
    grains = smooth(grains,3);
end
