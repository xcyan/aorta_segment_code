% =====================================
% CNN feedforward inference
% =====================================

function h = cnn_infer2(h1, weights, params)


batchsize = size(x, 4);

%numch == 1

hidhidlr = zeros(params.ws2, params.ws2, params.numhid2, params.numch);

for c = 1:params.numch,
    hidhidlr(:,:,:,c) = reshape(weights.hidhid(end:-1:1, end:-1:1, c,
    :),[params.ws2,params.ws2,params.numhid2]); %end:-1:1 flips the images because using valid conv
end

hbiasmat = repmat(permute(weights.hidhidbias,[2 3 1]),[size(x,1)-params.ws2+1,
size(x,2)-params.ws2+1, batchsize]);
hbiasmat = reshape(hbiasmat, [size(x,1)-params.ws2+1, size(x,2)-params.ws2+1,
params.numhid2, batchsize]);


h = hbiasmat;
if 0,
    % slow
    for c = 1:params.numch,
        for d = 1:params.numhid,
            for n = 1:batchsize,
                h(:,:,d,n) = h(:,:,d,n) + conv2(h1(:,:,c,n), hidhidlr(:,:,d,c), 'valid');
            end
        end
    end
else
    % fast
    for c = 1:params.numch,
        for d = 1:params.numhid2,
            h(:,:,d,:) = h(:,:,d,:) + convn(h1(:,:,c,:), hidhidlr(:,:,d,c), 'valid');
        end
    end
end

%use relu as activatin for everything besides output
switch params.nonlinearity,
    case 'relu',
        h = max(0, h);
    case 'sigmoid',
        h = sigmoid(h);
end


return;
