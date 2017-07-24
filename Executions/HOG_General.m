%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% -faces: 
% -images: Structure containing the images in .data(:,:,:,j) and the labels
%   of the images in .labels
% -cellStart: Start of the number of cells for the table
% -cendEnd: End of the number of cells for the table
% -nBinsStart: Start of number of bins for the table
% -nBinsEnd: End of number of bins for the table
% -CVO: The partitions for test and train

% Output:
% -myMatrixHOG: Table with the HOG with the given number of neighbours and
%   radius (the rows are the radius and the columns are the number of
%   neighbours).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function myMatrixHOG = HOG_General(faces, images, cellStart, cellEnd, nBinsStart, nBinsEnd, CVO)
    cellSize = cellStart;
    nBins = nBinsStart;
    
    myMatrixHOG = zeros((cellEnd - cellStart)+1, (nBinsEnd - nBinsStart)+1);
 
    for zx = 1:((nBinsEnd - nBinsStart)+1)
        for zy = 1:((cellEnd - cellStart)+1)
            % Extract HOG and SIFT featur es
            for i=1:length(faces)
                %i;
                % Obtain selected image
                if (length(size(images.data))>3)
                    % Color images
                    img = images.data(:,:,:,i);
                else
                    % Grayscale images
                    img = images.data(:,:,i);
                end
                
                % Extract HOG features
                [featureVector,hogVisualization] = ...
                    extractHOGFeatures(img, 'CellSize', [cellSize cellSize], ...
                    'NumBins', nBins);
                faces{i}.HOG = featureVector;
            end;
            
            disp(strcat('HOG          cellSize: ', int2str(cellSize), ...
                'x', int2str(cellSize), ' nBins: ', int2str(nBins)));
            
            errHOG = zeros(CVO.NumTestSets, 1);

            for i = 1:CVO.NumTestSets
                trIdx = CVO.training(i);
                teIdx = CVO.test(i);

                TrHOG = zeros(sum(trIdx), length(faces{1}.HOG));
                TeHOG = zeros(sum(teIdx), length(faces{1}.HOG));
                
                tr = 0;
                te = 0;
                
                for j = 1:length(faces)
                    if (teIdx(j)>0)
                        te = te + 1;
                        TeHOG(te,:) = faces{j}.HOG;
                    else
                        tr = tr + 1;
                        TrHOG(tr,:) = faces{j}.HOG;
                    end;
                end;

                t = templateSVM( 'Standardize', 1 );
                Mdl = fitcecoc(TrHOG, images.labels(trIdx), 'Learners', t);
                ytestHOG = predict(Mdl, TeHOG);

                errHOG(i) = sum(ytestHOG~=images.labels(teIdx)');

            end;

            % Calculate MAE
            myMatrixHOG(zy, zx) = sum(errHOG)/sum(CVO.TestSize);

            disp(strcat('MAE of       HOG:    ', sprintf('%f', myMatrixHOG(zy, zx))))
            
            % Newline
            disp(' ')
            
            % Go to the next element of the table
            cellSize = cellSize + 1;
        end;
        cellSize = cellStart;
        nBins = nBins + 1;
    end;
end