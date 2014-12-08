% =====================================
% CNN feedforward inference
% =====================================

function h = cnn_infer(x, weights, params)


batchsize = size(x, 4);

%numch == 1

vishidlr = zeros(params.ws, params.ws, params.numhid, params.numch);
for c = 1:params.numch,
    vishidlr(:,:,:,c) = reshape(weights.vishid(end:-1:1, end:-1:1, c, :),[params.ws,params.ws,params.numhid]); %end:-1:1 flips the images because using valid conv
end

hbiasmat = repmat(permute(weights.hidbias,[2 3 1]),[size(x,1)-params.ws+1, size(x,2)-params.ws+1, batchsize]);
hbiasmat = reshape(hbiasmat, [size(x,1)-params.ws+1, size(x,2)-params.ws+1, params.numhid, batchsize]);


h = hbiasmat;

for c = 1:params.numch,
    for d = 1:params.numhid,
        h(:,:,d,:) = h(:,:,d,:) + gather(convn(gpuArray(x(:,:,c,:)), gpuArray(vishidlr(:,:,d,c)), 'valid'));
    end
end


h = sigmoid(h);



return;
