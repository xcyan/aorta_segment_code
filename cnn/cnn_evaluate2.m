function [rec, prec, ap] = cnn_evaluate2(x, y, weights, params)

h = cnn_infer(x, weights, params);
h2 = cnn_infer2(h, weights, params);
yhat = cnn_recon(h2, weights, params);

[rec, prec, ap] = compute_ap(yhat(:), y(:));

return;
