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
% -CVO: The partitions for test and train

% Output:
% -myMatrixLBPPyramid: Table with the LBP with the given number of neighbours and
%   radius (the rows are the radius and the columns are the number of
%   neighbours).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function myMatrixLBPPyramid = LBP_of_pyramid_General(nlevels, blocksize, faces, images, nneighStart, nneighEnd, rStart, rEnd, CVO)
    nneighbours = nneighStart;
    radius = rStart;
    
    myMatrixLBPPyramid = zeros((nneighEnd - nneighStart)+1, (rEnd - rStart)+1); 

    myfaces = faces;
    for zx = 1:((rEnd - rStart)+1)
        for zy = 1:((nneighEnd - nneighStart)+1)         
            disp(strcat('LBP pyramid  nNeighbours: ', int2str(nneighbours), ...
                ' radius: ', int2str(radius)));
            
            for i=1:length(myfaces)
                %i
                % Obtain selected image
                img = images.data(:,:,:,i); 
                % Extract LBP features
                levels = pyramid_levels(nlevels, blocksize, img);
                lbpimg = LBP_of_pyramid(levels, nneighbours, radius);

                myfaces{i}.LBP = lbpimg;
            end;
            
            errLBP = zeros(CVO.NumTestSets, 1);
            
            for i = 1:CVO.NumTestSets
                trIdx = CVO.training(i);
                teIdx = CVO.test(i);
                
                TrLBP = zeros(sum(trIdx), length(myfaces{1}.LBP));
                TeLBP = zeros(sum(teIdx), length(myfaces{1}.LBP));
                
                tr = 0;
                te = 0;
    
                for j = 1:length(myfaces)
                    if (teIdx(j)>0)
                        te = te + 1;
                        TeLBP(te,:) = myfaces{j}.LBP;
                    else
                        tr = tr + 1;
                        TrLBP(tr,:) = myfaces{j}.LBP;
                    end;
                end;
    
                t = templateSVM( 'Standardize', 1 );
                Mdl = fitcecoc(TrLBP, images.labels(trIdx), 'Learners', t);
                ytestLBP = predict(Mdl, TeLBP);
    
                errLBP(i) = sum(ytestLBP~=images.labels(teIdx)');
            end;
            
            % Calculate MAE
            myMatrixLBPPyramid(zy, zx) = sum(errLBP)/sum(CVO.TestSize);
            
            disp(strcat('MAE of       LBP:    ', sprintf('%f', myMatrixLBPPyramid(zy, zx))))
            
            % Newline
            disp(' ')
            
            % Go to the next element of the table
            nneighbours = nneighbours + 2^(zy);
        end;
        nneighbours = nneighStart;
        radius = radius +1;
    end;
end