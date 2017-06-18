%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% -faces: structure with image labels etc
% -images: structure with the data of the images at images.data

% Output:
% -newfaces: structure with the data of the half image, and the data 
%   of the quarter image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function newfaces = resize_half_and_quart(faces, images)
    newfaces = cell(1, 1192);
    for i=1:length(faces)
        %i
        img = images.data(:,:,:,i);
        half = imresize(img, 0.5);
        quart = imresize(img, 0.25);
        newfaces{i}.half = half;
        newfaces{i}.quarter = quart;
    end;
end