function [ accuracy ] = test_cnn( test_data, test_labels, weights, filterInfo )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    [~, theta] = roll_params(0, weights);
    [accuracy, ~] = gradient_cnn(theta, test_data, test_labels, filterInfo, 0);
    
    

end

