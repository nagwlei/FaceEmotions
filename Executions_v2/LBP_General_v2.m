%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% -faces: Structure containing the emotion, etnicity, id etc.
% -images: Structure containing the images in .data(:,:,:,j) and the labels
%   of the images in .labels
% -cellSizeStart: Start of the number cell size
% -cellSizeEnd: End of the number of the cell size
% -nneighbours: Number of neighbours
% -radius
% -CVO: The partitions for test and train

% Output:
% -myArrayLBP: Array the MAE error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function myArrayLBP = LBP_General_v2(faces, images, cellSizeStart, cellSizeEnd, nneighbours, radius, CVO)
    
    cellSize = cellSizeStart;
    myArrayLBP = zeros(1, ((cellSizeEnd-cellSizeStart)+1));
    
    for zx=1:((cellSizeEnd-cellSizeStart)+1)
        % Extract LBP features
        for i=1:length(faces)
            if (length(size(images.data))>3)
                % Color image
                img = images.data(:,:,:,i);

                % Extract LBP features
                faces{i}.LBP = extractLBPFeatures(rgb2gray(uint8(img)), ...
                    'Upright',false, 'CellSize', [cellSize cellSize], ...
                    'NumNeighbors',nneighbours,'Radius',radius);
            else
                % Grayscale image
                img = images.data(:,:,i);

                % Extract LBP features
                faces{i}.LBP = extractLBPFeatures(uint8(img), ...
                    'Upright',false, 'CellSize', [cellSize cellSize], ...
                    'NumNeighbors',nneighbours,'Radius',radius);
            end

        end;

        disp(strcat('LBP          nNeighbours: ', int2str(nneighbours), ...
            ' radius: ', int2str(radius), '   cellSize: ', int2str(cellSize)));

        errLBP = zeros(CVO.NumTestSets, 1);

        for i = 1:CVO.NumTestSets
            trIdx = CVO.training(i);
            teIdx = CVO.test(i);

            % This is necessary to work with the CVO created for JAFFE
            if (strcmp(class(trIdx), 'cell'))
               trIdx = trIdx{1}; 
            end
            if (strcmp(class(teIdx), 'cell'))
                teIdx = teIdx{1};
            end

            TrLBP = zeros(sum(trIdx), length(faces{1}.LBP));
            TeLBP = zeros(sum(teIdx), length(faces{1}.LBP));

            tr = 0;
            te = 0;

            for j = 1:length(faces)
                if (teIdx(j)>0)
                    te = te + 1;
                    TeLBP(te,:) = faces{j}.LBP;
                else
                    tr = tr + 1;
                    TrLBP(tr,:) = faces{j}.LBP;
                end;
            end;

            t = templateSVM( 'Standardize', 1 );
            Mdl = fitcecoc(TrLBP, images.labels(trIdx), 'Learners', t);
            ytestLBP = predict(Mdl, TeLBP);

            errLBP(i) = sum(ytestLBP~=images.labels(teIdx)');
        end;

        % Introduce MAE in the matrix
        myArrayLBP(1, zx) = sum(errLBP)/sum(CVO.TestSize);

        disp(strcat('MAE of       LBP:    ', sprintf('%f', myArrayLBP(1, zx))))
        
        % Go to the next element of the array
        cellSize = cellSize + 1;
    end
end