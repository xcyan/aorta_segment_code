function [c] = cnn_cost(weights, xtrain, ytrain, params) 
  batchsize = size(xtrain, 4);
  weights = cnn_unroll2(weights, params); 
  % -- feed-forward inference
  h1 = cnn_infer(xtrain, weights, params);
  h2 = cnn_infer2(h1, weights, params);
  yhat = cnn_recon(h2, weights, params);

  % -- compute cost
  c = cross_entropy(ytrain, yhat)/batchsize;


end
