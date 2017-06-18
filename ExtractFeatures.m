run matlab/vl_compilenn

urlwrite('http://www.vlfeat.org/matconvnet/models/vgg-face.mat', 'vgg-face.mat');
net = load('vgg-face.mat');
load('ImageData\myfaces.mat');
load('ImageData\imdb.mat');

l = length(faces);

featTable = [];
for i=1:l
    i
    im = images.data(:,:,:,l);
    im_ = imresize(single(im), net.meta.normalization.imageSize(1:2));

    im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
    res = vl_simplenn(net, im_);
    feat4K = reshape(res(end-2).x, 4096, 1);
    features = feat4K / norm(feat4K, 2);
    trfeature = features';
    
    featTable = vertcat(featTable, trfeature);
end;
