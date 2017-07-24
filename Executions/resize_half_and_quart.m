%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% -faces: structure with image labels etc
% -images: structure with the data of the images at images.data

% Output:
% -newfaces: structure with the data of the half image, and the data 
%   of the quarter image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function newfaces = resize_half_and_quart(faces, images)
    newfaces = cell(1, length(faces));
    for i=1:length(faces)
        %i
        if (length(size(images.data))>3)
            % Color image
            img = images.data(:,:,:,i);
        else
            % Grayscale image
            img = images.data(:,:,i);
        end
        half = imresize(img, 0.5);
        quart = imresize(img, 0.25);
        newfaces{i}.half = half;
        newfaces{i}.quarter = quart;
    end;
end