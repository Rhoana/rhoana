function edge_volume = get_overlap_edges(volsize, volclass, cube_side)

%  cube_side is one of 1,2,3,4,5,6

LEFT    = 1; RIGHT   = 2;
UP      = 3; DOWN    = 4;
FRONT   = 5; BACK    = 6;

try
    
    edge_volume = zeros(volsize, volclass);
    %Mark edges with 1 for from/A edge and 2 for to/B edge
    switch cube_side
        case LEFT  % x=0
            edge_volume(:,1,:) = 2;
            edge_volume(:,end,:) = 1;
        case RIGHT % x=end
            edge_volume(:,1,:) = 1;
            edge_volume(:,end,:) = 2;
        case UP    % z=0
            edge_volume(:,:,1) = 2;
            edge_volume(:,:,end) = 1;
        case DOWN  % z=end
            edge_volume(:,:,1) = 1;
            edge_volume(:,:,end) = 2;
        case BACK  % y=0
            edge_volume(1,:,:) = 2;
            edge_volume(end,:,:) = 1;
        case FRONT % y=end
            edge_volume(1,:,:) = 1;
            edge_volume(end,:,:) = 2;
    end
    
catch me
    keyboard;
end


end