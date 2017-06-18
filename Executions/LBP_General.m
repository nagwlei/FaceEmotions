%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% -faces: 
% -images: Structure containing the images in .data(:,:,:,j) and the labels
%   of the images in .labels
% -nneighStart: Start of the number of neighbours for the table
% -nneighEnd: End of the number of neighbours for the table
% -rStart: Start of radius for the table
% -rEnd: End of radius for the table (and each of the radius will be 
%   calculated as a ^2)
% -nFolds: The number of folds to do the CrossValidation

% Output:
% -myMatrixLBP: Table with the LBP with the given number of neighbours and
%   radius (the rows are the radius and the columns are the number of
%   neighbours).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function myMatrixLBP = LBP_General(faces, images, nneighStart, nneighEnd, rStart, rEnd, nfolds)
    % LBPx: nneighbours
    % LBPy: Radius
    
    nneighbours = nneighStart;
    radius = rStart;
    
    
    for zx = 1:((rEnd - rStart)+1)
        for zy = 1:((nneighEnd - nneighStart)+1)
            % Extract LBP features
            for i=1:length(faces)
                img = images.data(:,:,:,i);
                % Extract LBP features
                lbpimg = extractLBPFeatures(rgb2gray(uint8(img)),'Upright',false, 'CellSize', [16 16], 'NumNeighbors',nneighbours,'Radius',radius);
                faces{i}.LBP = lbpimg;
            end;
            
            % Create the folds
            CVO = cvpartition(images.labels, 'k', nfolds);
            %CVO = cvpartition(images.labels, 'k', 10);
            errLBP = zeros(CVO.NumTestSets, 1);
            
            for i = 1:CVO.NumTestSets
                TrLBP = [];
                TsLBP = [];

                trIdx = CVO.training(i);
                teIdx = CVO.test(i);

                for j = 1:length(faces)
                    %j
                    if (teIdx(j)>0)
                        %disp('IF');
                        TsLBP = vertcat(TsLBP, faces{j}.LBP);
                    else
                        %disp('ELSE');
                        TrLBP = vertcat(TrLBP, faces{j}.LBP);
                    end;
                end;
    
                Mdl = fitcecoc(TrLBP, images.labels(trIdx));
                ytestLBP = predict(Mdl, TsLBP);

                errLBP(i) = sum(ytestLBP~=images.labels(teIdx)');
            end;
            
            cvErrLBP = sum(errLBP)/sum(CVO.TestSize);
            
            myMatrixLBP(zy, zx) = cvErrLBP;
            
            disp(strcat('VUELTA                zx:', int2str(zx), '    zy:', int2str(zy)))
            %disp(strcat('VALORRRRRRR                :    ', sprintf('%f', cvErrHOG)))
            disp(strcat('VALOR             LBP:    ', sprintf('%f', cvErrLBP)))
            % Go to the next element of the table
            nneighbours = nneighbours + 2^(zy);
            disp(strcat('CellSize  after     :', int2str(nneighbours)));
        end;
        nneighbours = 2;
        radius = radius +1;
    end;
end