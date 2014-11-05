function [cost, grad] = cnn_grad_roll(theta, xtrain, ytrain, params, xval, yval)


batchsize = size(xtrain, 4);
weights = cnn_unroll(theta, params);

% -- feed-forward inference
h1 = cnn_infer(xtrain, weights, params);
h2 = cnn_infer2(h1, weights, params);
yhat = cnn_recon(h, weights, params);


% -- compute cost
cost = cross_entropy(ytrain, yhat)/batchsize;


grad = replicate_struct(weights, 0);
% -- backprop

% objective -> h2
% get dh3
dobj = (yhat - ytrain)/batchsize;

% get db3
grad.visbias = grad.visbias + sum(dobj, 4);
% get dw3
for b = 1:size(weights.hidvis, 4),
    for c = 1:size(weights.hidvis, 3),
        grad.hidvis(:,:,c,b) = grad.hidvis(:,:,c,b) + convn(dobj(:,:,b,:), h2(end:-1:1,end:-1:1,c,end:-1:1), 'valid');
    end
end

% h2 -> h1

% get dh2
dh2 = zeros(size(h2));
for b = 1:size(weights.hidhid, 4),
  for c = 1:size(weights.hidhid, 3),
    dh2(:,:,c,:) = convn(dobj(:,:,b,:), weights.hidvis(end:-1:1,end:-1:1,c,b), 
      'valid');
  end
end

% get db2
grad.hid2bias = grad.hid2bias + permute(asdfasdf); % figure this out
% get dw2
for b = 1:size(weights.hidhid, 4)
    for c = 1:size(weights.hidhid, 3)
        grad.hidhid(:,:,c,b) = grad.hidhid(:,:,c,b) + convn(h1(:,:,c,:),
          dh2(end:-1:1,end:-1:1,c,end:-1:1), 'valid');
    end 
end

% h1 --> input 
% get dh1
dh1 = zeros(size(h1));
for b = 1:size(weights.hidvis, 4),
    for c = 1:size(weights.hidvis, 3),
        dh1(:,:,c,:) = convn(dh2(:,:,b,:), weights.hidhid(:,:,c,b), 'full');
    end
end

switch params.nonlinearity,
    case 'relu',
        dh1 = dh1.*(h1 > 0);
    case 'sigmoid',
        dh1 = dh1.*h1.*(1-h1);
end

% get db1
grad.hid1bias = grad.hid1bias + permute(sum(sum(sum(dobj, 1), 2), 4), [3 1 2]);
% get dw1
for b = 1:size(weights.vishid, 4),
    for c = 1:size(weights.vishid, 3),
        grad.vishid(:,:,c,b) = grad.vishid(:,:,c,b) + convn(xtrain(:,:,c,:), dh1(end:-1:1,end:-1:1,b,end:-1:1), 'valid');
    end
end

grad = cnn_roll(grad);


% -- evaluate
if exist('xval', 'var'),
    [~, ~, ap] = cnn_evaluate(xval, yval, weights, params);
    fprintf('val AP = %g\n', ap);
end


return;

