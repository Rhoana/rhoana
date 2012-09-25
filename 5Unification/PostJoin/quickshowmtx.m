function quickshowmtx (samples)

%Display a 3D image matrix as a movie of 2D images
colormap(gray);
ma = max(samples(:));
mi = min(samples(:));

a = size(samples,1);
d = sqrt(a);

x = 10;
y = 15;

nimg = 1;
first = 1;

pages = ceil(size(samples,2)/(x*y));
page = 1;

while page <= pages
    %while nimg <= size(samples,2)

    nimg = 1+(page-1)*x*y;
    displaygrid = ones(x*d+x+1, y*d+y+1) .* mi;

    %Show images in groups of x*y
    for j = 0:y-1
        for i = 0:x-1
            start_x = i*(d+1)+2;
            start_y = j*(d+1)+2;
            imagesc(reshape(samples(:,nimg), d, d));%pause;
            displaygrid(start_x:start_x+d-1, start_y:start_y+d-1) = ...
                reshape(samples(:,nimg), d, d);
            imagesc(displaygrid);%pause;
            nimg = nimg + 1;
            if nimg > size(samples,2)
                break;
            end
        end
        if nimg > size(samples,2)
            break;
        end
    end

    imagesc(displaygrid(:,:,1), [mi ma]);
    axis image off
    pause;
        page = page + 1;

%     user_entry = input('Return to repeat, q to quit, n for next, p for prev >', 's');
%     if isempty(user_entry)
%         %Repeat
%     elseif user_entry(1) == 'q'
%         break;
%     elseif user_entry(1) == 'n' && page < pages
%         page = page + 1;
%     elseif user_entry(1) == 'p' && page > 1
%         page = page - 1;
%     end

    %pause;

end