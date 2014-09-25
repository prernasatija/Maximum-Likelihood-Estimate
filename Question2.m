Training = load('training.txt')
Testing = load('testing.txt')
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

[numrows1,numcolumns1] = size(Training)
[numrows2,numcolumns2] = size(Testing)
ClassifyTrainingEqualP = [Training]
ClassifyTestingEqualP  = [Testing]
%Dimension of covariance matrix
d=2
%Initializing error count for training
Error_Count_Training = 0
i=0
for i=(1:numrows1)
    %Posterior Probability  is the product of multivariate probability density and prior
    %probability
    Posterior_Probability_1 = (1/3)*(exp(-0.5*(Training(i,1:2)-Mean_1)*inv(Covariance_1)*transpose(Training(i,1:2)-Mean_1)))/(((2*pi)^(d/2))*(det(Covariance_1))^(1/2))
    Posterior_Probability_2 = (1/3)*(exp(-0.5*(Training(i,1:2)-Mean_2)*inv(Covariance_2)*transpose(Training(i,1:2)-Mean_2)))/(((2*pi)^(d/2))*(det(Covariance_2))^(1/2))
    Posterior_Probability_3 = (1/3)*(exp(-0.5*(Training(i,1:2)-Mean_3)*inv(Covariance_3)*transpose(Training(i,1:2)-Mean_3)))/(((2*pi)^(d/2))*(det(Covariance_3))^(1/2))
    Posterior_Probabilities = [Posterior_Probability_1,Posterior_Probability_2,Posterior_Probability_3]
    Max_posterior_probability = max(Posterior_Probabilities)
    if Max_posterior_probability == Posterior_Probability_1
        ClassifyTrainingEqualP(i,4) = 1
    else if  Max_posterior_probability == Posterior_Probability_2
             ClassifyTrainingEqualP(i,4) = 2
        else
             ClassifyTrainingEqualP(i,4) = 3
        end
    end
    if Training(i,3) ~= ClassifyTrainingEqualP(i,4)
        Error_Count_Training = Error_Count_Training + 1
    end
end
Error_Rate_training = Error_Count_Training/numrows1
save('ClassifyTrainingEqualP.txt' ,'ClassifyTrainingEqualP', '-ascii');

%Initializing error count for Testing Data
Error_Count_Testing = 0
i=0

%Use the same mean and covariance for Testing data set
for i=(1:numrows2)
Posterior_Probability_1 = (1/3)*(exp(-0.5*(Testing(i,1:2)-Mean_1)*inv(Covariance_1)*transpose(Testing(i,1:2)-Mean_1)))/(((2*pi)^(d/2))*(det(Covariance_1))^(1/2))
Posterior_Probability_2 = (1/3)*(exp(-0.5*(Testing(i,1:2)-Mean_2)*inv(Covariance_2)*transpose(Testing(i,1:2)-Mean_2)))/(((2*pi)^(d/2))*(det(Covariance_2))^(1/2))
Posterior_Probability_3 = (1/3)*(exp(-0.5*(Testing(i,1:2)-Mean_3)*inv(Covariance_3)*transpose(Testing(i,1:2)-Mean_3)))/(((2*pi)^(d/2))*(det(Covariance_3))^(1/2))
Posterior_Probabilities = [Posterior_Probability_1,Posterior_Probability_2,Posterior_Probability_3]
Max_posterior_probability = max(Posterior_Probabilities)
    if Max_posterior_probability == Posterior_Probability_1
        ClassifyTestingEqualP(i,4) = 1
    else if  Max_posterior_probability == Posterior_Probability_2
             ClassifyTestingEqualP(i,4) = 2
        else
             ClassifyTestingEqualP(i,4) = 3
        end
    end
    if Testing(i,3) ~= ClassifyTestingEqualP(i,4)
        Error_Count_Testing = Error_Count_Testing + 1
    end  
end 
Error_Rate_testing = Error_Count_Testing/numrows2

save('ClassifyTestingEqualP.txt' , 'ClassifyTestingEqualP' ,'-ascii');
