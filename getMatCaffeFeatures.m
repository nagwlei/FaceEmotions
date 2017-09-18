function facesWithDeepFeatures = getMatCaffeFeatures( model_dir, model_name, weights_name, mean_file, gpu, faces, images )

    if gpu 
        caffe.set_mode_gpu();
        gpu_id = 0;  % we will use the first gpu in this demo
        caffe.set_device(gpu_id);
    else
        caffe.set_mode_cpu();
    end
    
    net_weights = [model_dir weights_name];
    net_model = [model_dir model_name];
    
    phase = 'test'; % run with phase test (so that dropout isn't applied)
    if ~exist(net_weights, 'file')
      error('Please download the model from its site before you run this script');
    end

    % Initialize a network
    net = caffe.Net(net_model, net_weights, phase);

    % caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
    % is already in W x H x C with BGR channels
    mean_data = caffe.io.read_mean( mean_file );
    CROPPED_DIM = 224;
    mean_data_resized = imresize(mean_data,[CROPPED_DIM CROPPED_DIM], 'bilinear');

    tic;

    if (length(size(images.data))>3)
        colorImages = 1;
    else
        colorImages = 0;
    end
                    
    
    for kk = 1:length( faces )
        %disp( kk );
        % apply network to image

        % read image
        if( colorImages )
            im = images.data(:,:,:, kk);
        else
            im = images.data(:,:,kk);
            im = cat(3, im, im, im);
        end
        % Convert an image returned by Matlab's imread to im_data in caffe's data
        % format: W x H x C with BGR channels
        im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
        im_data = permute(im_data, [2, 1, 3]);  % flip width and height
        im_data = single(im_data);  % convert from uint8 to single


        im_data = imresize(im_data, [CROPPED_DIM CROPPED_DIM], 'bilinear');  % resize im_data
        im_data = im_data - mean_data_resized;  % subtract mean_data (already in W x H x C, BGR)
        crops_data=im_data;

        input_data = { crops_data };

        %tic;
        % The net forward function. It takes in a cell array of N-D arrays
        % (where N == 4 here) containing data of input blob(s) and outputs a cell
        % array containing data from output blob(s)
        scores = net.forward(input_data);
        %toc;

        fc7 = net.blobs('fc7').get_data();
        %fc6 = net.blobs('fc6').get_data();
       
        % apply L2 norm to features
        faces{ kk }.deep_features = fc7 / norm( fc7, 2 );

    end

    toc;
    
    facesWithDeepFeatures = faces;
    
    % call caffe.reset_all() to reset caffe
    caffe.reset_all();

end