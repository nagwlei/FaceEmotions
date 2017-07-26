%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% -faces: Structure containing the emotion, etnicity, id etc.
% -images: Structure containing the images in .data(:,:,:,j) and the labels
%   of the images in .labels
% -nneighStart: Start of the number of neighbours for the table
% -nneighEnd: End of the number of neighbours for the table  (and each of 
%   the neighbours will be calculated as a ^2)
% -rStart: Start of radius for the table
% -rEnd: End of radius for the table
% -CVO: The partitions for test and train

% Output:
% -myMatrixLBP: Table with the LBP with the given number of neighbours and
%   radius (the rows are the number of neighbours and the columns are 
%   the radius).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function myMatrixLBP = LBP_General(faces, images, nneighStart, nneighEnd, rStart, rEnd, CVO)
    nneighbours = nneighStart;
    radius = rStart;
    
    myMatrixLBP = zeros((nneighEnd - nneighStart)+1, (rEnd - rStart)+1);
    
    for zx = 1:((rEnd - rStart)+1)
        for zy = 1:((nneighEnd - nneighStart)+1)
            % Extract LBP features
            for i=1:length(faces)
                if (length(size(images.data))>3)
                    % Color image
                    img = images.data(:,:,:,i);
                    
                    % Extract LBP features
                    faces{i}.LBP = extractLBPFeatures(rgb2gray(uint8(img)), ...
                        'Upright',false, 'CellSize', [16 16], ...
                        'NumNeighbors',nneighbours,'Radius',radius);
                else
                    % Grayscale image
                    img = images.data(:,:,i);
                    
                    % Extract LBP features
                    faces{i}.LBP = extractLBPFeatures(uint8(img), ...
                        'Upright',false, 'CellSize', [16 16], ...
                        'NumNeighbors',nneighbours,'Radius',radius);
                end
                
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
            
            % Introduce MAE in the matrix
            myMatrixLBP(zy, zx) = sum(errLBP)/sum(CVO.TestSize);
            
            disp(strcat('MAE of       LBP:    ', sprintf('%f', myMatrixLBP(zy, zx))))

            % Newline
            disp(' ')
            
            % Go to the next element of the table
            nneighbours = nneighbours + 2^(zy);
        end;
        nneighbours = nneighStart;
        radius = radius +1;
    end;
end