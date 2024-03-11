MATLAB code for calculation of angularly resolved chord length distribution in polycrystalline microstructures. 

## Contents

`calc_CLD.m` - main functions for AR-CLD calculation

`plot_CLD.m` - utility function to visualize AR-CLD

`gb2gbs.m` - utility function that provides grain boundary segments (needed for `calc_CLD`) for a given [grain boundary class](https://mtex-toolbox.github.io/files/doc/BoundaryAnalysis.html) obtained in [MTEX](https://mtex-toolbox.github.io/files/doc/BoundaryAnalysis.html).

## Example Usage with MTEX

```MATLAB
%% <load EBSD data to variable ebsd using MTEX>
%% <clean-up if necessary>

% segment grains in MTEX, using, say 10 deg threshold
grains = calcGrains(ebsd,'angle',10*degree);

% get grain boundary segments
gbs = gb2gbs(grains.boundary);

%% set calculation settings
% spacing between test lines
dxy = max(ebsd.unitCell) - min(ebsd.unitCell);
spacing = min(dxy);

% list of directions in deg
angs = 0:1:179;

% chord max in histogram
cho_max = 1.2*max(grains.diameter);

%% main calculation
% calculate AR-CLD
stats = calc_CLD(gbs,spacing,angs,'cho_max',cho_max,'silent');

% visualize AR-CLD for 360 deg
cld_full = [stats.CLD;stats.CLD(2:end-1,:)];
plot_CLD(stats.hst_x,cld_full)
```

## Publication

To cite this code and see more details, refer to

M.I. Latypov, M. Kühbach, I.J. Beyerlein, J.-C. Stinville, L.S. Toth, T.M. Pollock, S.R. Kalidindi, [Application of chord length distribution and principal component analysis of diverse polycrystalline microstructures](https://doi.org/10.1016/j.matchar.2018.09.020), Materials Characterization, In Press (2018).
