%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
  
% Ouputs:
% -LBParray: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function LBParray = LBP_of_pyramid(pyramid, LBPx, LBPy)
    % Auxiliary to calculate the size of LBParray
    syms k;
    nblocks = symsum(k^2, k, 1, length(pyramid));
    nblocks = double(nblocks);
    LBParray = zeros(1, nblocks);
    
    % Loop through the levels of the pyramid
    for i=1:length(pyramid)
        % Loop through the images inside each level
        for j=1:(length(pyramid{i})^2)
            % Obtain the LBP features of each image and concat it to output
            % array
            LBPfeat = extractLBPFeatures(pyramid{i}{j},'Upright',false, 'CellSize', [16 16], 'NumNeighbors',LBPx,'Radius',LBPy);
            LBParray = horzcat(LBParray, LBPfeat);
        end;
    end;
end