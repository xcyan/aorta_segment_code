function [cost, grad] = cnn_grad_roll_2_layer(theta, xtrain, ytrain, params, xval, yval)


batchsize = size(xtrain, 4);
weights = cnn_unroll2(theta, params);

% -- feed-forward inference
h1 = cnn_infer(xtrain, weights, params);
h2 = cnn_infer2(h1, weights, params);
yhat = cnn_recon(h2, weights, params);


% -- compute cost
cost = cross_entropy(ytrain, yhat)/batchsize;


grad = replicate_struct(weights, 0);
% -- backprop

% objective -> h2
% get dh3
dobj = (yhat - ytrain)/batchsize;

% get db3
grad.visbias = sum(dobj, 4);
% get dw3
for b = 1:size(weights.hidvis, 4),
    for c = 1:size(weights.hidvis, 3),
        grad.hidvis(:,:,c,b) = grad.hidvis(:,:,c,b) + convn(dobj(:,:,b,:), h2(end:-1:1,end:-1:1,c,end:-1:1), 'valid');
    end
end

% h2 -> h1

% get dh2
dh2 = zeros(size(h2));
for b = 1:size(weights.hidvis, 4),
  for c = 1:size(weights.hidvis, 3),
    dh2(:,:,c,:) = convn(dobj(:,:,b,:), weights.hidvis(end:-1:1,end:-1:1,c,b),'valid');
  end
end

dh2 = dh2.*h2.*(1-h2);

% get db2
%grad.hid2bias = grad.hid2bias + permute(sum(sum(sum(dh2, 1), 2), 4), [3 1 2]); % figure this out
biasSum = sum(sum(sum(dh2,1),2),4);
grad.hid2bias = biasSum(:);
% get dw2
for b = 1:size(weights.hidhid, 4)
    for c = 1:size(weights.hidhid, 3)
        grad.hidhid(:,:,c,b) = grad.hidhid(:,:,c,b) + convn(h1(:,:,c,:),dh2(end:-1:1,end:-1:1,b,end:-1:1),'valid');
    end 
end

% h1 --> input 
% get dh1
dh1 = zeros(size(h1));
for b = 1:size(weights.hidhid, 4),
    for c = 1:size(weights.hidhid, 3),
        % Changed this according to new derivations
        dh1(:,:,b,:) = dh1(:,:,b,:) + convn(dh2(:,:,c,:), weights.hidhid(:,:,b,c), 'full');
    end
end

dh1 = dh1.*h1.*(1-h1);

% get db1
%grad.hidbias = grad.hidbias + permute(sum(sum(sum(dh1, 1), 2), 4), [3 1 2]);
biasSum = sum(sum(sum(dh1, 1),2),4);
grad.hidbias = biasSum(:);
% get dw1
for b = 1:size(weights.vishid, 4),
    for c = 1:size(weights.vishid, 3),
        grad.vishid(:,:,c,b) = grad.vishid(:,:,c,b) + convn(xtrain(:,:,c,:), dh1(end:-1:1,end:-1:1,b,end:-1:1), 'valid');
    end
end

%FOR TESTING
%grad.hidbias = zeros(size(grad.hidbias));
%grad.hid2bias = zeros(size(grad.hid2bias));
%grad.visbias = zeros(size(grad.visbias));
%END FOR TESTING

grad = cnn_roll2(grad);

roll_weight = cnn_roll2(weights);

%gradient_checking(@(x) cnn_cost(x, xtrain, ytrain, params), roll_weight, grad); 

% -- evaluatexval
if exist('xval', 'var'),
    [~, ~, ap] = cnn_evaluate(xval, yval, weights, params);
    fprintf('val AP = %g\n', ap);
end


return;


