%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% -nlevels: The number of levels of the pyramid is suppose to be
% -blocksize: The size of the blocks of the image
% -image: The image in which the PML is going to be applied

% Ouput:
% -levels = Cell array containing in each of the cells (which represent the
%   levels) with the image divided in level^2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function levels = pyramid_levels(nlevels, blocksize, image)
    % Cell array with the image divided in k^k cells in level k 
    levels = cell(1, nlevels);
    for i=1:nlevels
        newimage = imresize(image, [i*blocksize i*blocksize]);
        
        % Auxiliary variable to do mat2cell
        x = [];
        for j=1:i
            x = horzcat(x, 16);
        end;
        
        % Images need to be in grayscale for mat2cell (and LBP too)
        gray = newimage;
        C = mat2cell(gray, x, x);
        % Output variable containing in each cell a set of cells with the
        % subimages
        levels{i} = C;
    end;
end