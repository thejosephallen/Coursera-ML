function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

# X is 5000x401 and Theta1 is 25x401
a2 = sigmoid([ones(m, 1) X] * Theta1');
# a2 is 5000x25 and Theta 2 is 10x26
a3 = sigmoid([ones(m, 1) a2] * Theta2');
# a3 is 5000x10

#stores the max prediction over the labels for each example
[max_label, max_index] = max(a3');

# the prediction vector (5000x1) containing nums(1-10)
p = max_index';







% =========================================================================


end
