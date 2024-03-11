function gbs = gb2gbs(gb)
% return GB segments given MTEX GB object
% input: MTEX GB object, e.g., gb = grains.boundary
% output: 4xN array containing GB segments 

    % end points boundary segments
    gbs(1,:) = gb.V(gb.F(:,1),1).';
    gbs(2,:) = gb.V(gb.F(:,1),2).';
    gbs(3,:) = gb.V(gb.F(:,2),1).';
    gbs(4,:) = gb.V(gb.F(:,2),2).';

end