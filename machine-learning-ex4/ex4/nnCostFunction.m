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


% reformating the y for vetorized implimentaion. ie, eg,  1 changed to [1,0,0,0,0,0,0,0,0,0]
Y=zeros(size(y,1),num_labels);
for i=1:size(y,1)
    switch(y(i))
        case 1
            Y(i,1)=1;
        case 2
            Y(i,2)=1;
        case 3
            Y(i,3)=1;
        case 4
            Y(i,4)=1;
        case 5
            Y(i,5)=1;
        case 6
            Y(i,6)=1;
        case 7
            Y(i,7)=1;
        case 8
            Y(i,8)=1;
        case 9
            Y(i,9)=1;
        case 10
            Y(i,10)=1;    
    endswitch;
endfor;

X=[ones(m,1),X];

% cost, a2,a3( h(x) )  calculation  using forward propagation 

a2=sigmoid(X*Theta1');
a2=[ones(m,1),a2];
hx=sigmoid(a2*Theta2');
fl=((Y.*log(hx))+((1-Y).* log(1-hx) ));
ksum=sum(fl,2);
J=((-1)/m)*sum(ksum);

% add regularization to the cost;

theta1Reg=Theta1(:,2:end);
theta2Reg=Theta2(:,2:end);

regValue= (lambda/(2*m))*(sum(sum(theta1Reg.^2,2))+sum(sum(theta2Reg.^2,2)));

J=J+regValue;



%delta 3 -- error associated with output nodes. ie the direct difference between the hypothisis and the the actual output 
delta3=hx-Y;

% calculating delata2 using backpropagation 

sg_z2= a2.*(1-a2);
delta2=(Theta2' * delta3')' .* sg_z2;





% calculating DELTA2 for each dataSet 
% todo try to vectorize to avoid the looping 

Delta2 = zeros(size(Theta2));
for i=1:m
    Delta2= Delta2+ (delta3(i,:)'*a2(i,:)); 
endfor;

% calculating DELTA1 
% Also removed the first column of delata2

Delta1=zeros(size(Theta1));
for i=1:m
    Delta1= Delta1+ (delta2(i,2:end)'*X(i,:)); 
endfor;


% unregularised gradient of J(theta) with respect to theta(i,j)

Theta1_grad= (1/m)* Delta1;

Theta2_grad= (1/m)* Delta2;


% regularizing the gradinet 

Theta1_grad(:,2:end)= Theta1_grad(:,2:end)+ (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end)= Theta2_grad(:,2:end)+ (lambda/m)*Theta2(:,2:end);


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
