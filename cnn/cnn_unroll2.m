function weights = cnn_unroll2(theta, params)

idx = 0;

weights.vishid = reshape(theta(idx+1:idx+params.ws^2*params.numch*params.numhid), ...
    params.ws, params.ws, params.numch, params.numhid);
idx = idx + numel(weights.vishid);

weights.hidbias = theta(idx+1:idx+params.numhid);
idx = idx + numel(weights.hidbias);

weights.hidhid = reshape(theta(idx+1:idx+params.ws2^2*params.numch*params.numhid*params.numhid2), ... 
    params.ws2, params.ws2, params.numhid, params.numhid2);
idx = idx + numel(weights.hidhid);

weights.hid2bias = theta(idx+1:idx+params.numhid2);
idx = idx + numel(weights.hid2bias);

weights.hidvis = reshape(theta(idx+1:idx+params.ws3^2*params.numhid2*params.numout), ...
    params.ws3, params.ws3, params.numhid2, params.numout);
idx = idx + numel(weights.hidvis);

weights.visbias = reshape(theta(idx+1:idx+params.rs*params.cs*params.numout), params.rs, params.cs, params.numout);
idx = idx + numel(weights.visbias);

assert(idx == length(theta));


return;