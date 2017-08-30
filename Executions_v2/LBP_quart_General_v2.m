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
% -myArrayLBPquart: Array the MAE error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function myArrayLBPquart = LBP_quart_General_v2(ffaces, newfaces, images, ...
    cellSizeStart, cellSizeEnd, nneighbours, radius, CVO)
    
    cellSize = cellSizeStart;
    
    myArrayLBPquart = zeros(1, ((cellSizeEnd - cellSizeStart)+1));

     for zx = 1:((cellSizeEnd - cellSizeStart)+1)
        % Extract LBP features
        for i=1:length(faces)
            %i;
            % Obtain selected image
            if (length(size(images.data))>3)
                aux = images.data(:,:,:,i);
                img = rgb2gray(uint8(aux));
                half = rgb2gray(uint8(newfaces{i}.half));
                quart = rgb2gray(uint8(newfaces{i}.quarter));
            else
                aux = images.data(:,:,i);
                img = uint8(aux);
                half = uint8(newfaces{i}.half);
                quart = uint8(newfaces{i}.quarter);
            end

            % Extract LBP features
            lbpimg = extractLBPFeatures(img,'Upright',false, ...
                'CellSize', [cellSize cellSize], 'NumNeighbors', ...
                nneighbours,'Radius',radius);

            lbphalfimg = extractLBPFeatures(half,'Upright',false, ...
                'CellSize', [cellSize cellSize], 'NumNeighbors',nneighbours, ...
                'Radius',radius);

            lbpquartimg = extractLBPFeatures(quart,'Upright',false, ...
                'CellSize', [cellSize cellSize], 'NumNeighbors',nneighbours, ...
                'Radius',radius);
            finallbp = horzcat(lbpimg, lbphalfimg, lbpquartimg);
            faces{i}.LBP = finallbp;
        end;

        disp(strcat('LBP          nNeighbours: ', int2str(nneighbours), ...
            ' radius: ', int2str(radius)));

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

        myMatrixLBPquart(zy, zx) = sum(errLBP)/sum(CVO.TestSize);

        disp(strcat('MAE of       LBP:    ', sprintf('%f', myMatrixLBPquart(zy, zx))))

        % Newline
        disp(' ')

        % Go to the next element of the array
        cellSize = cellSize + 1;
    end;
end