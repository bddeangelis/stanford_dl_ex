function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei); % Theta values are an aggregate of all the W and b elements
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1); % cell to store activations at each level
gradStack = cell(numHidden+1, 1); % cell to store the gradients of each level

keyboard;


%% forward prop
%%% YOUR CODE HERE %%%
% Forward propagation's purpose is to give a hypothesis value for each example for comparison to ground truth

% % % Define the batch size
% % batchSize = 1000;
% % start = 1; % starting example

% % Initialize the first layer of activations
% hAct{1} = data;

% Loop through each of the layers
for ii = 1:size(stack,1)
    
    if ii == 1
        
        % Expand the bias unit to be the appropriate number of examples
        bias = repmat(stack{ii}.b, 1, size(data,2));
            
        % Calculate the activations for each neuron
        hAct{ii} = sigmoid(stack{ii}.W * data + bias);
        
    else
        
        % Expand the bias unit to be the appropriate number of examples
        bias = repmat(stack{ii}.b, 1, size(hAct{ii-1},2));
        
        % Calculate the activations for each neuron
        hAct{ii} = sigmoid(stack{ii}.W * hAct{ii-1} + bias); 
    
    end
%     % Given the input, calculate the activation for each neuron
%     hAct{ii+1} = sigmoid(z(ii+1));
    
%     % Alternative tanh
%     a{ii+1} = ...
%     % Alternative rectified linear
%     a{ii+1} = ...

end

keyboard;

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

% Generate a groundTruth table using the labels data
 % Isolate the hypothesized probabilities corresponding to the correct labels
  I=sub2ind(size(hAct{end}), labels', 1:size(hAct{end},2));
  labelH = hAct{end}(I);
  
  % Generate a truth table associated with where the labels in y are correct
  truthTable = zeros(size(hAct{end}));
  truthTable(I) = 1;
  
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

% gradStack =

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

% weight norm cost
% wCost = 

% Sum together the components of the cost
% cost = ceCost + wCost;




%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



