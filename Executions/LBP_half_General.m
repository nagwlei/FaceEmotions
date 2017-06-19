%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% -faces: 
% -newfaces: structure containing the half and the quarter iamges
% -images: Structure containing the images in .data(:,:,:,j) and the labels
%   of the images in .labels
% -nneighStart: Start of the number of neighbours for the table
% -nneighEnd: End of the number of neighbours for the table
% -rStart: Start of radius for the table
% -rEnd: End of radius for the table (and each of the radius will be 
%   calculated as a ^2)
% -nFolds: The number of folds to do the CrossValidation

% Output:
% -myMatrixLBPhalf: Table with the LBP with the given number of neighbours and
%   radius (the rows are the radius and the columns are the number of
%   neighbours).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function myMatrixLBPhalf = LBP_half_General(faces, newfaces, images, nneighStart, nneighEnd, rStart, rEnd, nfolds)
    % LBPx: nneighbours
    % LBPy: Radius

    nneighbours = nneighStart;
    radius = rStart;

    myMatrixLBPhalf = zeros((nneighEnd - nneighStart)+1, (rEnd - rStart)+1);

     for zx = 1:((rEnd - rStart)+1)
         for zy = 1:((nneighEnd - nneighStart)+1)
            % Extract LBP features
            for i=1:length(faces)
                %i;
                % Obtain selected image
                img = images.data(:,:,:,i);
                % Extract LBP features
                lbpimg = extractLBPFeatures(rgb2gray(uint8(img)),'Upright',false, 'CellSize', [16 16], 'NumNeighbors',nneighbours,'Radius',radius);
                half = newfaces{i}.half;
                lbphalfimg = extractLBPFeatures(rgb2gray(uint8(half)),'Upright',false, 'CellSize', [16 16], 'NumNeighbors',nneighbours,'Radius',radius);
                finallbp = horzcat(lbpimg, lbphalfimg);
                faces{i}.LBP = finallbp;
            end;

            % Create the folds
            CVO = cvpartition(images.labels, 'k', 5);
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


            myMatrixLBPhalf(zy, zx) = cvErrLBP;


            disp(strcat('VUELTA                zx:', int2str(zx), '    zy:', int2str(zy)))
            %disp(strcat('VALORRRRRRR                :    ', sprintf('%f', cvErrHOG)))
            disp(strcat('VALOR             LBP:    ', sprintf('%f', cvErrLBP)))
            % Go to the next element of the table
            nneighbours = nneighbours + 2^(zy);
            disp(strcat('CellSize  after     :', int2str(nneighbours)));
        end;
        nneighbours = nneighStart;
        radius = radius +1;
    end;
end
