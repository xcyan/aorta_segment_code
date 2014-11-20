function weights = cnn_train_lbfgs_2_layer(xtrain, ytrain, params, xval, yval)


% -- initialization
rng(1)
weights = struct;
weights.vishid = 0.01*randn(params.ws, params.ws, params.numch, params.numhid);
weights.hidhid = 0.01*randn(params.ws2, params.ws2, params.numhid, params.numhid2)
weights.hidvis = 0.01*randn(params.ws3, params.ws3, params.numhid2, params.numout);
weights.hidbias = zeros(params.numhid, 1);
weights.hid2bias = zeros(params.numhid2, 1);
weights.visbias = zeros(params.rs, params.cs, params.numout);

weights.vishid

addpath(genpath('utils/minFunc_2012/'));
theta = cnn_roll2(weights);


% lbfgs
options.method = 'lbfgs';
options.maxiter = params.maxiter;


opttheta = minFunc(@(p) cnn_grad_roll_2_layer(p, xtrain (:,:,:,1:3), ytrain(:,:,:,1:3), params, xval, yval), theta, options);
weights = cnn_unroll2(opttheta, params); 


% -- filename to save
fname_mat = sprintf('models/%s.mat', params.fname);
save(fname_mat, 'weights', 'params');


return;
