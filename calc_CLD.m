function [stats,chords] = calc_CLD(gbs,spacing,ang_range,varargin)
% Calculate angularly resolved chord length distribution as described in
% Latypov et al., Materials Characterization (2018)
%
% Usage: 
%   stats = calc_CLD(gbs,spacing,ang_range)
%
% INPUT:
% - gbs is a 4xN array of boundary segments: [x1,y1,x2,y2]
% - spacing is the spacing between test lines
% - ang_range is a list of CLD sampling directions (in deg.)
% OUTPUT:
% - structure array with calculated statistics
%
% Marat I. Latypov (2018)
% latmarat@ucsb.edu

    % default settings
    exclude = 1; 
    silent = 0;
    show = 0;
    vectorize = 0;
    set_cho_max = NaN;
    cho_nbins = 100;
    tiny_i = 0;
    
    %% digest input 
    if nargin > 3
        for ii=1:length(varargin)
            if strcmp(varargin{ii},'include') == 1
                exclude = 0;
                disp('Intersections with bounding box will be considered')
            end
            if strcmp(varargin{ii},'silent') == 1
                silent = 1;
                disp('Runtime messages supressed')
            end
            if strcmp(varargin{ii},'show') == 1
                show = 1;
                disp('Figure requested')
            end
            if strcmp(varargin{ii},'vectorize') == 1
                vectorize = 1;
                disp('Vectorized computation requested')
            end
            if strcmp(varargin{ii},'cho_max') == 1
                set_cho_max = varargin{ii+1};
            end
            if strcmp(varargin{ii},'cho_nbins') == 1
                cho_nbins = varargin{ii+1};
            end
            if strcmp(varargin{ii},'min_cl') == 1
                tiny_i = varargin{ii+1};
            end
        end
    end

    % convert angles to radians
    degree = pi/180.0;
    ang_range = -ang_range*degree;

    %% Initial Geometry
    
    % get bounding box for initial domain
    xmin = min(min([gbs(1,:),gbs(3,:)]));
    xmax = max(max([gbs(1,:),gbs(3,:)]));
    ymin = min(min([gbs(2,:),gbs(4,:)]));
    ymax = max(max([gbs(2,:),gbs(4,:)]));
    
    % side lengths
    Lx = xmax-xmin;
    Ly = ymax-ymin;
    
    % center data
    cntr = [Lx/2;Ly/2;Lx/2;Ly/2];
    gbs_0 = gbs - repmat(cntr,[1,size(gbs,2)]);
    
    % get bounding box for centered domain
    xmin = min(min([gbs_0(1,:),gbs_0(3,:)]));
    xmax = max(max([gbs_0(1,:),gbs_0(3,:)]));
    ymin = min(min([gbs_0(2,:),gbs_0(4,:)]));
    ymax = max(max([gbs_0(2,:),gbs_0(4,:)]));
    
    if ~silent
        fprintf('Corners of centered domain: (%.2f,%.2f), (%.2f,%.2f)\n',...
            xmin,ymin,xmax,ymax);
    end

    %% find abs min and max of domain for all rotations
    dom = [[xmin;ymin],[xmin;ymax],[xmax;ymin],[xmax;ymax]];
    dom_rot = zeros([numel(ang_range),size(dom)]);
    for irot = 1:numel(ang_range)
        theta = ang_range(irot);
        rot = [cos(theta) -sin(theta); sin(theta) cos(theta)];
        dom_rot(irot,:,:) = rot*dom;   
    end
    abs_x_min = min(min(dom_rot(:,1,:)));
    abs_x_max = max(max(dom_rot(:,1,:)));
    abs_y_min = min(min(dom_rot(:,2,:)));
    abs_y_max = max(max(dom_rot(:,2,:)));
    
    %% test lines
    seed_y = (abs_y_min-spacing/2):spacing:(abs_y_max+spacing/2);
    % seed_x1 = repmat(abs_x_min-spacing,1,numel(seed_y));
    % seed_x2 = repmat(abs_x_max+spacing,1,numel(seed_y));

    %% histogram settings
    if isnan(set_cho_max)
        cho_range = [0,max([Lx/2.0,Ly/2.0])];
    else
        cho_range = [0,set_cho_max];
    end
    hst_dx = (cho_range(2)-cho_range(1))/cho_nbins;
    hst_ed = cho_range(1):hst_dx:cho_range(2);
    hst_x = hst_ed(1:end-1) + diff(hst_ed)/2;
    
    % tiny length threshold
    tiny = min([Lx,Ly])/1e6;
    if tiny_i ~= 0
        tiny=tiny_i;
    end
    fprintf('Chords less than %.2e will be neglected\n',tiny);

    %% main calculation
    % allocation
    cho_max_all = zeros([numel(ang_range),1]);
    cld = zeros([numel(ang_range),cho_nbins]);
    hst_f_all = zeros([numel(ang_range),cho_nbins]);
    
    tic
    % loop over rotations
    for irot = 1:numel(ang_range)
        % current rotation angle
        theta = ang_range(irot);
        if theta == 0 
            gbs = gbs_0;
        else 
            % get current end points
            bs_1 = gbs_0(1:2,:)';
            bs_2 = gbs_0(3:4,:)';

            % rotation matrix
            rot = [cos(theta) -sin(theta); sin(theta) cos(theta)];

            % rotate segments
            gbs = [bs_1*rot', bs_2*rot']';
        end
        
        % get new bounding box
%         xmin = min(min([gbs(:,1),gbs(:,3)]));
%         xmax = max(max([gbs(:,1),gbs(:,3)]));
        ymin = min(min([gbs(2,:),gbs(4,:)]));
        ymax = max(max([gbs(2,:),gbs(4,:)]));

        % get active test lines
        ind = seed_y <= ymax & seed_y >= ymin;
        seed_x1 = repmat(abs_x_min-spacing,1,numel(seed_y(ind)));
        seed_x2 = repmat(abs_x_max+spacing,1,numel(seed_y(ind)));
        probe_1 = [seed_x1;seed_y(ind)]';
        probe_2 = [seed_x2;seed_y(ind)]';
        
        if ~silent
            fprintf('Number of test lines: %d\n',nnz(ind))
        end
        
        % plot GB segments
        if show
            figure;
            line([gbs(1,:);gbs(3,:)],[gbs(2,:);gbs(4,:)],'Color',[85,85,85]/255,'LineWidth',1);
            hold all
            line([probe_1(:,1)';probe_2(:,1)'],[probe_1(:,2)';probe_2(:,2)'],'Color',[30,30,255]/255,'LineWidth',1.25);
            axis equal; axis off;
        end
        
        if vectorize
            % get chords and intersection points
            [chords,x_all,y_all] = calc_chords(gbs,probe_1,probe_2,exclude);
            
            % remove tiny chord lengths
            ntiny = nnz(chords<tiny);
            chords(chords<tiny) = [];
            
            % bin chord lengths
            [hst_f,~] = histcounts(chords,hst_ed);
            
            % find the max chord length for this direction
            cho_max_all(irot) = max(chords);
            
            % visualize intersection points
            if show
                scatter(x_all(:),y_all(:),25,'MarkerFaceColor',[255,30,30]/255,'MarkerEdgeColor',[255,30,30]/255,'MarkerFaceAlpha',0.5);
            end
        else
            % allocate
            hst_f = zeros([1,cho_nbins]);
            cho_max = zeros([1,numel(seed_y(ind))]);
            ntiny = 0;

            % loop over test lines
            for ii = 1:numel(seed_y(ind))
                % get chords and intersection points              
                idx_1 = gbs(2,:) > probe_1(ii,2) & gbs(4,:) < probe_1(ii,2);
                idx_2 = gbs(4,:) > probe_1(ii,2) & gbs(2,:) < probe_1(ii,2);
                
                [chords,x,y] = calc_chords(gbs(:,idx_1|idx_2),probe_1(ii,:),probe_2(ii,:),exclude);
                
                % remove tiny chord lengths
                ntiny = ntiny + nnz(chords<tiny);
                chords(chords<tiny) = [];
                
                % bin chord lengths and get max chord 
                if ~isempty(chords)
                    [hst_N,~] = histcounts(chords,hst_ed);
                    hst_f = hst_f + hst_N;
                    cho_max(ii) = max(chords);
                else
                    cho_max(ii) = 0;
                end
                
                % plot test line and intersections
                if show
                    % plot GB segments intersecting the test lines
                    line([gbs(1,idx_1|idx_2);gbs(3,idx_1|idx_2)],[gbs(2,idx_1|idx_2);gbs(4,idx_1|idx_2)],'Color',[30,255,30]/255,'LineWidth',2);
                    % plot the test lines
                    line([probe_1(ii,1)';probe_2(ii,1)'],[probe_1(ii,2)';probe_2(ii,2)'],'Color',[30,30,255]/255);
                    % plot the test lines
                    scatter(x(:),y(:),25,'MarkerFaceColor',[255,30,30]/255,'MarkerEdgeColor',[255,30,30]/255,'MarkerFaceAlpha',0.5);
                end
%                 if ~silent
%                     fprintf('Done with %d out of %d test lines',ii,numel(seed_y(ind)));
%                 end
            end
            cho_max_all(irot) = max(cho_max);
        end
        
        % store the results into AR-CLD as a pdf
        hst_f_all(irot,:) = hst_f;
        cld(irot,:) = hst_x.*hst_f/(sum(hst_x.*hst_f));
        
        % print runtime message
        if mod(theta,30.0*degree) == 0 && theta ~= 0 && ~silent
            fprintf('Done with angle %.2f\n',theta/degree)
            fprintf('Number of neglected tiny chords: %d\n',ntiny);
        end
    end
    
    % store results
    cho_abs_max = max(cho_max_all);
    stats = struct(...
        'hst_x',hst_x,...
        'hst_f',hst_f_all,...
        'CLD',cld,...
        'abs_max_chord',cho_abs_max,...
        'angle_range',ang_range,...
        'chord_range',cho_range,...
        'no_bins',cho_nbins,...
        'hst_dx',hst_dx);
    toc
end

function [chords,x,y] = calc_chords(gbs,probe_1,probe_2,exclude)
    % Get chord lengths for given test lines and GB segments
    % as described in Latypov et al. Materials Characterization (2018)
    %
    % Usage:
    %   [x,y] = calc_chords(gbs,xy1,xy2)
    %
    % INPUT:
    %   gbs - 3xN array of boundary segments 
    %   xy1,xy2 - coordinates of the start and end points 
    %             of the test line
    % OUTPUT:
    %  x,y - coordinates of intersection points
    %     
    % Marat I. Latypov (2018)
    % latmarat@ucsb.edu
    %
    % Acknowledgments  
    %   Hielsher (https://github.com/mtex-toolbox/mtex)

    % get intersection points
    [x,y] = calc_inters(gbs,probe_1,probe_2);
    
    % sort x coordinates of intersections
    x_sort = sort(x,2);
    
    % find x deltas
    x_diff = diff(x_sort,1,2);
    
    % remove rows with all NaNs 
    % corresponds to test lines not intersecting segments
    x_diff(~any(~isnan(x_diff), 2),:)=[];

    % rm intersections with bounding box
    if exclude 
        bnd_idx = sum(~isnan(x_diff),2);
        x_diff(sub2ind(size(x_diff),1:size(x_diff,1),bnd_idx')) = NaN;
        x_diff(:,1) = NaN;
    end

    % get chord lengths as x deltas
    chords = x_diff(~isnan(x_diff));
    
end

function  [x,y] = calc_inters(gbs,xy1,xy2)
    % Get intersection points between boundary segments and test lines
    %
    % Usage:
    %   [x,y] = calc_inters(gbs,xy1,xy2)
    %
    % INPUT:
    %   gbs - 3xN array of boundary segments 
    %   xy1,xy2 - coordinates of the start and end points 
    %             of the test line
    % OUTPUT:
    %  x,y - coordinates of intersection points
    % 
    % Acknowledgments: 
    %   Bourke (http://paulbourke.net/geometry/pointlineplane/)
    %   Hielsher (https://github.com/mtex-toolbox/mtex)
    
    
    n_rows_1 = size(xy1,1);
    n_rows_2 = size(gbs,2);

    % end points of the lines
    X1 = repmat(xy1(:,1),1,n_rows_2);
    Y1 = repmat(xy1(:,2),1,n_rows_2);
    X2 = repmat(xy2(:,1),1,n_rows_2);
    Y2 = repmat(xy2(:,2),1,n_rows_2);

    X3 = repmat(gbs(1,:),n_rows_1,1);
    X4 = repmat(gbs(3,:),n_rows_1,1);
    Y3 = repmat(gbs(2,:),n_rows_1,1);
    Y4 = repmat(gbs(4,:),n_rows_1,1);
    
    clear gbs;
    
    X4_X3 = X4-X3; clear X4
    Y1_Y3 = Y1-Y3; 
    Y4_Y3 = Y4-Y3; clear Y3 Y4;
    X1_X3 = X1-X3; clear X3;
    X2_X1 = X2-X1; clear X2;
    Y2_Y1 = Y2-Y1; clear Y2;

    numerator_a = X4_X3 .* Y1_Y3 - Y4_Y3 .* X1_X3;
    numerator_b = X2_X1 .* Y1_Y3 - Y2_Y1 .* X1_X3;
    denominator = Y4_Y3 .* X2_X1 - X4_X3 .* Y2_Y1;
    
    clear X4_X3 Y4_Y3;
    
    u_a = numerator_a ./ denominator;
    u_b = numerator_b ./ denominator;
    inside = (u_a >= 0) & (u_a <= 1) & (u_b >= 0) & (u_b <= 1);

    x = X1 + X2_X1 .* u_a;
    y = Y1 + Y2_Y1 .* u_a;
    
    clearvars -except x y inside;

    x(~inside) = NaN;
    y(~inside) = NaN;

end
