function overlap_volume = get_cube_overlap(cube_volume, cube_ix, cube_side, dicing_pars)

%  cube_side is one of 1,2,3,4,5,6

LEFT    = 1; RIGHT   = 2;
UP      = 3; DOWN    = 4;
FRONT   = 5; BACK    = 6;

switch cube_side
    
    case LEFT  % x=0
        step = 1;
        xi = (dicing_pars.minX(cube_ix)+1:dicing_pars.maxX(cube_ix - step)-1) - dicing_pars.minX(cube_ix) + 1;
        overlap_volume = cube_volume(:, xi, :);

    case RIGHT % x=end
        step = 1;
        xi = (dicing_pars.minX(cube_ix + step)+1:dicing_pars.maxX(cube_ix)-1) - dicing_pars.minX(cube_ix) + 1;
        overlap_volume = cube_volume(:, xi, :);

    case UP    % z=0
        step = dicing_pars.max_x_index * dicing_pars.max_y_index;
        zi = (dicing_pars.minZ(cube_ix):dicing_pars.maxZ(cube_ix - step)) - dicing_pars.minZ(cube_ix) + 1;
        overlap_volume = cube_volume(:, :, zi);

    case DOWN  % z=end
        step = dicing_pars.max_x_index * dicing_pars.max_y_index;
        zi = (dicing_pars.minZ(cube_ix + step):dicing_pars.maxZ(cube_ix)) - dicing_pars.minZ(cube_ix) + 1;
        overlap_volume = cube_volume(:, :, zi);

    case BACK  % y=0
        step = dicing_pars.max_x_index;
        yi = (dicing_pars.minY(cube_ix)+1:dicing_pars.maxY(cube_ix - step)-1) - dicing_pars.minY(cube_ix) + 1;
        overlap_volume = cube_volume(yi, :, :);

    case FRONT % y=end
        step = dicing_pars.max_x_index;
        yi = (dicing_pars.minY(cube_ix + step)+1:dicing_pars.maxY(cube_ix)-1) - dicing_pars.minY(cube_ix) + 1;
        overlap_volume = cube_volume(yi, :, :);

end
    
end