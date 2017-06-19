%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% -faces: 
% -images: Structure containing the images in .data(:,:,:,j) and the labels
%   of the images in .labels
% -cellStart: Start of the number of cells for the table
% -cendEnd: End of the number of cells for the table
% -nBinsStart: Start of number of bins for the table
% -nBinsEnd: End of number of bins for the table
% -nFolds: The number of folds to do the CrossValidation

% Output:
% -myMatrixHOG: Table with the HOG with the given number of neighbours and
%   radius (the rows are the radius and the columns are the number of
%   neighbours).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function myMatrixHOG = HOG_General(faces, images, cellStart, cellEnd, nBinsStart, nBinsEnd, nfolds)
    %HOGx CellSize
    %HOGy NumBins
    
    cellSize = cellStart;
    nBins = nBinsStart;
    myMatrixHOG = zeros((cellEnd - cellStart)+1, (nBinsEnd - nBinsStart)+1);
 
    for zx = 1:((nBinsEnd - nBinsStart)+1)
        for zy = 1:((cellEnd - cellStart)+1)
            % Extract HOG and SIFT featur es
            for i=1:length(faces)
                %i;
                % Obtain selected image
                img = images.data(:,:,:,i);
                % Extract HOG features
                [featureVector,hogVisualization] = extractHOGFeatures(img, 'CellSize', [cellSize cellSize], 'NumBins', nBins);
                faces{i}.HOG = featureVector;
            end;

            % Create the folds
            CVO = cvpartition(images.labels, 'k', nfolds);
            %CVO = cvpartition(images.labels, 'k', 10);
            %CVO = cvpartition(images.set, 'k', 7);
            errHOG = zeros(CVO.NumTestSets, 1);

            for i = 1:CVO.NumTestSets
                TrHOG = [];
                TsHOG = [];
                trIdx = CVO.training(i);
                teIdx = CVO.test(i);

                for j = 1:length(faces)
                    %j
                    if (teIdx(j)>0)
                        %disp('IF');
                        TsHOG = vertcat(TsHOG, faces{j}.HOG);
                    else
                        %disp('ELSE');
                        TrHOG = vertcat(TrHOG, faces{j}.HOG);
                    end;
                end;

                Mdl2 = fitcecoc(TrHOG, images.labels(trIdx));
                ytestHOG = predict(Mdl2, TsHOG);

                errHOG(i) = sum(ytestHOG~=images.labels(teIdx)');

            end;
            cvErrHOG = sum(errHOG)/sum(CVO.TestSize);

            myMatrixHOG(zx, zy) = cvErrHOG; 

            disp(strcat('VUELTAAAAAAA                zx:', int2str(zx), '    zy:', int2str(zy)))
            disp(strcat('VALORRRRRRR                :    ', sprintf('%f', cvErrHOG)))
            % Go to the next element of the table
            cellSize = cellSize + 1;
        end;
        cellSize = cellStart;
        nBins = nBins + 1;
    end;
end