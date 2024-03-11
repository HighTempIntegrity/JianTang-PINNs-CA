function plot_CLD(xbin,f,varargin)

    flip = 0;
    clim = 0;
    set_theme = 0;
    set_ang = 1;
    if nargin > 2
        for ii=1:length(varargin)
            if strcmp(varargin{ii},'flip') == 1
                flip = 1;
            end
            if strcmp(varargin{ii},'clim') == 1
                clim = 1;
                lims = varargin{ii+1};
            end
            if strcmp(varargin{ii},'theme') == 1
                set_theme = 1;
                theme = varargin{ii+1};
            end
            if strcmp(varargin{ii},'in_ang') == 1
	        set_ang = 0;
	        ang_step = varargin{ii+1};
            end
        end
    end
    if set_ang
        ang_step = 0:2*pi/(size(f,1)-1):2*pi;
    end  
    [r,t] = meshgrid(xbin,ang_step);
    x = r.*cos(t);
    y = r.*sin(t);
    figure;
    [~,h] = contourf(x,y,f,128); axis image; axis off
    colorbar;
    set(h,'edgecolor','none');
    if set_theme && exist('brewermap','file')
        map = brewermap(128,theme);
        if flip 
            map = flipud(map);
        elseif strcmp(theme,'Spectral') || strcmp(theme,'RdBu')
            fprintf('Warning: Spectral and RdBu usually need to be flipped\n');
        end
        colormap(map)
    end
    
    % set limits
    if clim
        caxis(lims);
    end
end
    
