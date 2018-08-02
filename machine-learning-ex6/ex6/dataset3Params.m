function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

min_error = inf;

for i=1:length(C_vec),
  C_test = C_vec(i);
  for j=1:length(sigma_vec),
    sigma_test = sigma_vec(j);
    
    % Train model with these C and sigma parameters
    model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));

    % Get this model's predictions on the CV set
    predictions = svmPredict(model, Xval);

    % Compute model's prediction error on the CV set     
    test_error = mean(double(predictions ~= yval));
    
    % Check if these parameters give lower error than previous values
    if (test_error < min_error),
      C = C_test;
      sigma = sigma_test;
      min_error = test_error;
    endif;
  end;
end;




% =========================================================================

end
