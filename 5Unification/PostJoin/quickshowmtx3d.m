function quickshowmtx3d (samples)

%Display a 3D image matrix as a movie of 2D images
colormap(gray);
ma = max(samples(:));
mi = min(samples(:));

zw = size(samples,3);

a = size(samples,1);
d = a;

x = 10;
y = 15;
if size(samples,4) <= 6
    x = 2;
    y = 3;
elseif size(samples,4) <= 24
    x = 4;
    y = 6;
elseif size(samples,4) <= 54
    x = 6;
    y = 9;
elseif size(samples,4) <= 96
    x = 8;
    y = 12;
end

nimg = 1;
first = 1;

pages = ceil(size(samples,4)/(x*y));
page = 1;

while page <= pages
    %while nimg <= size(samples,2)

    nimg = 1+(page-1)*x*y;
    displaygrid = ones(x*d+x+1, y*d+y+1, zw) .* mi;

    %Show images in groups of x*y
    for j = 0:y-1
        for i = 0:x-1
            start_x = i*(d+1)+2;
            start_y = j*(d+1)+2;
            
            %quickshowsc(samples(:,:,:,nimg));pause;
            displaygrid(start_x:start_x+d-1, start_y:start_y+d-1, :) = ...
                samples(:,:,:,nimg);
            nimg = nimg + 1;
            if nimg > size(samples,4)
                break;
            end
        end
        if nimg > size(samples,4)
            break;
        end
    end

    if first == 1
        imagesc(displaygrid(:,:,1), [mi ma]);
        axis image off
        pause;
        first = 0;
    end

    for z = 1:zw
        imagesc(displaygrid(:,:,z), [mi ma]);
        axis image off
        pause(0.1);
    end

    user_entry = input('\nReturn or r to repeat, q to quit, n for next, p for prev >', 's');
    if isempty(user_entry) || user_entry(1) == 'r'
        %Repeat
    elseif user_entry(1) == 'q'
        break;
    elseif user_entry(1) == 'n' && page < pages
        page = page + 1;
    elseif user_entry(1) == 'p' && page > 1
        page = page - 1;
    end

    %pause;

end