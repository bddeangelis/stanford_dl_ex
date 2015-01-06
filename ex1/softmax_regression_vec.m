function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

%   keyboard;
  
  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%

  % Append zeros to the theta values correponding to the zero'd class
  theta = [ theta, zeros(size(theta,1),1) ];
%   
%   % Calculate the hypothesized probabilities for each example
%   h = exp(theta'*X);
%   
%   % Normalize all hypothesized probabilities for each example by dividing by the sum of probabilities for each example
%   normH = bsxfun(@rdivide, h, sum(h,1));
%   
%   % Isolate the hypothesized probabilities corresponding to the correct labels
%   I=sub2ind(size(normH), y, 1:size(normH,2));
%   labelH = h(I);
%   
%   % Generate a truth table associated with where the labels in y are correct
%   truthTable = zeros(size(normH));
%   truthTable(I) = 1;
  
  % Try Dan Luu's way
  groundTruth = full(sparse(y, 1:m, 1));
%   groundTruth = groundTruth(1:end-1,:);
  td = theta' * X;
%   td = bsxfun(@minus, td, max(td));
  temp = exp(td);
  
    denominator = sum(temp);
    p = bsxfun(@rdivide, temp, denominator);
    f = (-1/m) * sum(sum(groundTruth .* log(p)));
    g = (-1/m) * (groundTruth - p) * X';
    g = g';
    
  
% %   % Calculate the cost function 
% %   % Are these supposed to be averages rather than sums? are we doing something in batch here?
% %   f = (-1/m)*sum(log(labelH));
% % %   f = -sum(log(labelH));
% %   
% %   % Calculate the gradient
% %   % Are these supposed to be averages rather than sums? are we doing something in batch here?
% %   g = (-1/m)*X*(truthTable-normH)';
% %   
  % Delete off the last column from g
  g = g(:,1:end-1);
  
  % Reshape the gradient into a vector  
  g=g(:); % make gradient a vector for minFunc
  
end

