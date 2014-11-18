function theta = cnn_roll2(weights)

theta = [];
theta = [theta ; weights.vishid(:)];
theta = [theta ; weights.hidbias(:)];
theta = [theta ; weights.hidhid(:)];
theta = [theta ; weights.hid2bias(:)];
theta = [theta ; weights.hidvis(:)];
theta = [theta ; weights.visbias(:)];

return;