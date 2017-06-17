%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% -nlevels: Number of levels of the pyramid
% -blocksize: The size of each of the blocks inside each of the levels
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
% -myMatrixLBPPyramid: Table with the LBP with the given number of neighbours and
%   radius (the rows are the radius and the columns are the number of
%   neighbours).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function myMatrixLBPPyramid = LBP_of_pyramid_General(nlevels, blocksize, faces, images, nneighStart, nneighEnd, rStart, rEnd, nfolds)
    LBPx = nneighStart;
    LBPy = rStart;
    
    myMatrixLBPPyramid = zeros((nneighEnd - nneighStart)+1, (rEnd - rStart)+1); 

    myfaces = faces;
    for zx = 1:((rEnd - rStart)+1)
        for zy = 1:((nneighEnd - nneighStart)+1)
            disp(strcat('zy: ', int2str(zy)));
            for i=1:length(myfaces)
                %i
                % Obtain selected image
                img = images.data(:,:,:,i); 
                % Extract LBP features
                levels = pyramid_levels(nlevels, blocksize, img);
                lbpimg = LBP_of_pyramid(levels, LBPx, LBPy);
                %lbpimg = extractLBPFeatures(rgb2gray(uint8(img)),'Upright',false, 'CellSize', [16 16], 'NumNeighbors',LBPx,'Radius',LBPy);
                myfaces{i}.LBP = lbpimg;
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
    
                for j = 1:length(myfaces)
                    %j
                    if (teIdx(j)>0)
                        %disp('IF');
                        TsLBP = vertcat(TsLBP, myfaces{j}.LBP);
                    else
                        %disp('ELSE');
                        TrLBP = vertcat(TrLBP, myfaces{j}.LBP);
                    end;
                end;
    
    
                Mdl = fitcecoc(TrLBP, images.labels(trIdx));
                ytestLBP = predict(Mdl, TsLBP);
    
                errLBP(i) = sum(ytestLBP~=images.labels(teIdx)');
            end;
            
            % Calculate MAE
            cvErrLBP = sum(errLBP)/sum(CVO.TestSize);
        
            myMatrixLBPPyramid(zy, zx) = cvErrLBP;
        
            disp(strcat('VUELTA                zx:', int2str(zx), '    zy:', int2str(zy)))
            %disp(strcat('VALORRRRRRR                :    ', sprintf('%f', cvErrHOG)))
            disp(strcat('VALOR             LBP:    ', sprintf('%f', cvErrLBP)))
            % Go to the next element of the table
            LBPx = LBPx + 2^(zy);
            disp(strcat('CellSize  after     :', int2str(LBPx)));
        end;
        LBPx = nneighStart;
        LBPy = LBPy +1;
    end;
end