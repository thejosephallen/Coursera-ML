function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


# The elements of y in vector form are the corresponding row y in the identity
# matrix formed by the number of classes
Y = eye(num_labels)(y,:); #5000x10


#///////////////// Part 1 - Feedforward and cost function ////////////////
# computes prediction for all x given theta parameters
a1 = [ones(m,1), X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);

z3 = [ones(m,1), a2] * Theta2';
a3 = sigmoid(z3);

h = a3; #5000x10

# Cost is the sum over all labels and over all examples
J = (1/m)*sum(sum(-Y.*log(h)-(1-Y).*log(1-h)));


# ----------- Regularized cost function -------------------------------

# Discard bias units
theta1 = Theta1(:, 2:size(Theta1,2)); #25x401 => 25x400
theta2 = Theta2(:, 2:size(Theta2,2)); #10x26 => 10x25 

# Add (lambda/2*m)*(summed squares of all non-bias theta) to cost 
J += (lambda/(2*m))*(sum(sum(theta1.^2)) + sum(sum(theta2.^2)));



#/////////////// Part 2 - Backpropagation //////////////////////////
D2 = 0;
D1 = 0;

for i = 1:m,
  # Forward propagate to find activations and hypothesis
  a1 = [1, X(i,:)]; #(1x401)
  z2 = a1 * Theta1'; # (1x401) x (401x25) = (1x25)
  a2 = [1, sigmoid(z2)];
  z3 = a2 * Theta2'; # (1x26) x (26x10) = (1x10)
  a3 = sigmoid(z3); # (1x10)
  
  # Compute the error for each node j in layer l and backpropagate it
  d3 = a3 - Y(i,:); # (1x10)
  d2 = d3 * theta2 .* sigmoidGradient(z2); # (1x10) x (10x25) = (1x25)

  # Accumulate gradients
  D2 += (d3' * a2); #(10x1) x (1x26) = (10x26)
  D1 += (d2' * a1); #(25x1) x (1x401) = (25x401)
  
end;

# 1/m times the accumulated gradients
Theta2_grad = (1/m)*D2; #(10x25)
Theta1_grad = (1/m)*D1; #(25x401)


% -------------------- Regularize the gradients ---------------------------
Theta2_grad(:, 2:end) += ((lambda/m) * theta2);
Theta1_grad(:, 2:end) += ((lambda/m) * theta1);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
