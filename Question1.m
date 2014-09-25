Training = load('training.txt')
X1= Training((Training(:,3)==1),1:2);
%Size function to get the number of rows belonging to class 1
[m1,n1] = size(X1)

% Maximum Likelihood Estimates for mean and covariance of class label 1
Mean_1 = sum(X1)/m1

Covariance_1 = cov(X1)

X2= Training((Training(:,3)==2),1:2);

%Size function to get the number of rows belonging to class 2
[m2,n2] = size(X2)

% Maximum Likelihood Estimates for mean and covariance of class label 2
Mean_2 = sum(X2)/m2
Covariance_2 = cov(X2)

X3= Training((Training(:,3)==3),1:2);
%Size function to get the number of rows belonging to class 3
[m3,n3] = size(X3)

% Maximum Likelihood Estimates for mean and covariance of class label 3
Mean_3 = sum(X3)/m3
Covariance_3 = cov(X3)