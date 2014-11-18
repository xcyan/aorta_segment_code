function [ yval_pred, ytest_pred, inf_time, paramsname ] = train_weights_2layercnn_sep( group_idx, xtrain, ytrain, xval, yval, xtest, alpha)
%TRAIN_WEIGHTS_1LAYERCNN_SEP Summary of this function goes here
%   Detailed explanation goes here

load('global_params.mat');

if ~exist('opt', 'var'),
    % optimization method, 'sgd' or 'lbfgs'
    opt = 'lbfgs';
end
if ~exist('numhid', 'var'),
    % number of convolutional filters in the second layer.
    numhid = 30;
end

if ~exist('numhid2', 'var'),
    % number of convolutional filters in the the third layer.
    numhid2 = 30;
end 

if ~exist('ws1', 'var'),
    % filter size
    ws = 8;
end

if ~exist('ws2', 'var'), 
    % 2nd layer filter size
    ws2 = 16;
end

if ~exist('ws3', 'var')
    % Output layer filter size
    ws3 = 23; %check this

if ~exist('optmask', 'var'),
    % post processing with mask
    optmask = 1;
end

if ~optmask,
    alpha = 0;
end

if alpha == 0,
    optmask = 0;
end

if ~exist('Flag_HU','var'),
    Flag_HU = 0;
end

if ~exist('Flag_caffe', 'var'),
    Flag_caffe = 0;
end

xtrain = permute(xtrain, [1 2 4 3]);
ytrain = permute(ytrain, [1 2 4 3]);
xval = permute(xval, [1 2 4 3]);
yval = permute(yval, [1 2 4 3]);
xtest = permute(xtest, [1 2 4 3]);

%% Normalizing the intensity
% if (Flag_HU),
%
% else
%     [x_m, x_stds] = fit_normal_distribution(cat(4, xtrain, xval, xtest));
%     xtrain = fit_normal_distribution(xtrain, x_m, x_stds);
%     xval = fit_normal_distribution(xval, x_m, x_stds);
%     xtest = fit_normal_distribution(xtest, x_m, x_stds);
%     save('nd_params.mat','x_m','x_stds');
% end

if optmask,
    mask_prior = mean(ytrain, 4);
    mask_prior = max(0, tanh(mask_prior*alpha));
end

% -- convNet params setting
params = struct(...
    'dataset', ['aorta_split_' sprintf('%02d',group_idx) ], ...
    'numch', size(xtrain, 3), ...
    'numhid', numhid, ...
    'numhid2', numhid2, ...
    'numout', 1, ...
    'optimize', opt, ...
    'ws', ws, ...
    'ws2', ws2, ...
    'ws3', ws3, ...
    'rs', gparams.mean_rsize, ...
    'cs', gparams.mean_csize, ...
    'nonlinearity', 'relu', ...
    'eps', 0.0001, ...
    'eps_decay', 0.01, ...
    'maxiter', 200, ...
    'batchsize', 10, ...
    'momentum_change', 0, ...
    'momentum_init', 0.33, ...
    'momentum_final', 0.5, ...
    'verbose', 1);

params = setfield(params, 'fname', sprintf('%s_%s_itr_%d', params.dataset, params.optimize, params.maxiter));
disp(params);
paramsname = params.fname;

% -- convNet learning (forward - backward)
MODELSRC = 'models/';
mkdir(MODELSRC);

if (Flag_caffe),
    
else
    if exist([MODELSRC sprintf('%s_al%g.mat', params.fname, alpha)], 'file'),
        load([MODELSRC sprintf('%s_al%g.mat', params.fname, alpha)], 'params', 'weights', 'w', 'b', 'mask_prior');
    else
        
        % Learn CNN weights
        if strcmp(params.optimize, 'sgd'),
            weights = cnn_train(xtrain, ytrain, params, xval, yval);
        elseif strcmp(params.optimize, 'lbfgs'),
            disp('Training with lbfgs');
            weights = cnn_train_lbfgs_2_layer(xtrain, ytrain, params, xval, yval);
            disp('Finished training');
        end
        % Learn additional classifier with mask prior
        if optmask,
            yhtrain = zeros(size(ytrain));
            for i = 1:size(xtrain, 4)
                xc = xtrain(:, :, :, i);
                
                % inference
                h = cnn_infer(xc, weights, params);
                yhat = cnn_recon(h, weights, params);
                yhtrain(:, :, :, i) = yhat.*mask_prior;
            end
            yhtrain = yhtrain(:);
    'eps', 0.0001, ...
            ytrain_flat = ytrain(:);
            [w, b] = logistic_regression(yhtrain', ytrain_flat', 0);
            clear yhtrain ytrain_flat;
            save([MODELSRC sprintf('%s_al%g.mat', params.fname, alpha)], 'params','weights','w','b', 'mask_prior');
        end
    end
end

yval_pred = zeros(size(xval));
ytest_pred = zeros(size(xtest));

for i = 1:size(xval, 4),
    xc = xval(:, :, :, i);
    
    h = cnn_infer(xc, weights, params);
    h2 = cnn_infer2(h, weights, params);
    yhat = cnn_recon2(h2, weights, params);
    
    if optmask,
        yhat = sigmoid(w*yhat.*mask_prior + b);
    end
    
    yval_pred(:, :, :, i) = yhat;
end

te = 0;
for i = 1:size(xtest, 4),
    xc = xtest(:, :, :, i);
    
    ts = tic;
    h = cnn_infer(xc, weights, params);
    h2 = cnn_infer2(h, weights, params);
    yhat = cnn_recon(h2, weights, params);
    'eps', 0.0001, ...
    te = te + toc(ts);
    
    if optmask,
        yhat = sigmoid(w*yhat.*mask_prior + b);
    end
    
    ytest_pred(:, :, :, i) = yhat;
end
inf_time = te/size(xtest, 4);


end
