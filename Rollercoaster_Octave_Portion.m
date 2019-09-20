#II. Normalizing Feature (Octave Source Code)
function [X_norm,mu,sd]=Normalization(X)
  X_norm=X;
  mu=zeros(1,size(X)(2));
  sd=zeros(1,size(X)(2));
  for i=1:size(X)(2)
    mu(1,i)=mean(X(:,i));
    sd(1,i)=std(X(:,i));
    X_norm(:,i)=(X(:,i).-mu(1,i))/sd(1,i);
  endfor
endfunction
#III. Gradient Descent for Linear Regression (Octave Source Code)
function [theta, J_history] = Gradient_Descent(X, y, theta, alpha, num_iters)
m=length(y); 
J_history=zeros(num_iters, 1);
for iter=1:num_iters
  h=X*theta;
  delta=h-y;
  theta(1)=theta(1)-(alpha/m)*sum(delta);
  theta(2:end)=theta(2:end)-(alpha/m)*sum(delta'*X(:,2:end),1)';
  J_history(iter) = ComputeCost(X, y, theta);
endfor
endfunction
#IV. Computing Cost Function of the Linear Regression Model (Octave Source Code)
function J=ComputeCost(X, y, theta)
m=length(y); 
J=0;
J=sumsq(X*theta-y)/(2*m);
endfunction
#V. Training the Linear Regression Model to predict Drop Time and Angle (Octave Source Code)
data=load('data.txt');
X=data(:,1:4);
Y=data(:,5);
[X_norm,X_mu,X_sigma]=Normalization(X); %Normalize the data
X_norm=[ones(size(X)(1),1) X_norm];
m=length(Y);
alpha=0.1;
num_iters=50000;
size(X);
theta=zeros(5,1);
perc=0.6
count=floor(perc*size(X)(1)); %divide the training and test set
[theta, J_history] = Gradient_Descent(X_norm(1:count,:),Y(1:count),theta,alpha,num_iters);
test_err=2*ComputeCost(X_norm(count+1:end,:),Y(count+1:end,:),theta)
#VI. Hypothesis Function of a model basing on s=vt (Octave Source Code)
function h=Model(X,theta)
  X1=((X(:,1)*theta(2))+(X(:,3)*theta(6))).+theta(3);
  X2=(X(:,2)*theta(4)).+theta(5);
  h=theta(1).+(X1./X2);
endfunction
#VII. Gradient Descent for a model basing on s=vt (Octave Source Code)
function [theta, J_history] = Gradient_Descent(X, y, theta, alpha, num_iters)
m=length(y); 
J_history=zeros(num_iters, 1);
for iter=1:num_iters
  h=X*theta;
  delta=h-y;
  theta(1)=theta(1)-(alpha/m)*sum(delta);
  theta(2:end)=theta(2:end)-(alpha/m)*sum(delta'*X(:,2:end),1)';
  J_history(iter) = ComputeCost(X, y, theta);
endfor
endfunction
#VIII. Training Model basing on s=vt to predict time (Octave Source Code)
data=load('data_time.txt');
X=data(:,1:3);
Y=data(:,4);
[X_norm,X_mu,X_sigma]=Normalization(X);
m=length(Y);
alpha=1.5;
num_iters=50000;
theta=ones(6,1);
perc=0.6
num=floor(perc*size(X)(1));
[theta,J_history]=Grad_Descent2(X_norm(1:num,:),Y(1:num,1),theta,alpha,num_iters);
test_err=2*ComputeCost2(X_norm(num+1:end,:),Y(num+1:end,1),theta)
