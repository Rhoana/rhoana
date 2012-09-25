function quickshow (img3)
%Display a 3D image as a movie of 2D images
if size(img3,4) > 1
    %Colour image
    for z = 1:size(img3,4)
        imshow(img3(:,:,:,z));
        pause(0.05);
    end
else
    for z = 1:size(img3,3)
        imshow(img3(:,:,z), 'InitialMagnification', 'fit');
        pause(0.1);
    end
end