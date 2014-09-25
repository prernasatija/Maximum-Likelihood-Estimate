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
% Prior Probability is the ratio of Number of samples of the class to Total
% Number of samples
Prior_Probability_1 = m1/numrows1
Prior_Probability_2 = m2/numrows1
Prior_Probability_3 = m3/numrows1

ClassifyTrainingMLEP =[Training]
ClassifyTestingMLEP  =[Testing]
%Dimension of covariance matrix
d=2
%Initializing class and error count for training data
Error_Count_Training = 0

for i=1:numrows1
    
    Probability_Density_Training_1 = (Prior_Probability_1)*(exp(-0.5*((Training(i,1:2)-Mean_1))*inv(Covariance_1)*(transpose(Training(i,1:2)-Mean_1)))/[((2*pi).^(d/2)*(det(Covariance_1)).^(1/2))])
    Probability_Density_Training_2 = (Prior_Probability_2)*(exp(-0.5*((Training(i,1:2)-Mean_2))*inv(Covariance_2)*(transpose(Training(i,1:2)-Mean_2)))/[((2*pi).^(d/2)*(det(Covariance_2)).^(1/2))]);
    Probability_Density_Training_3 = (Prior_Probability_3)*(exp(-0.5*((Training(i,1:2)-Mean_3))*inv(Covariance_3)*(transpose(Training(i,1:2)-Mean_3)))/[((2*pi).^(d/2)*(det(Covariance_3)).^(1/2))]);
    Probability_Densities = [Probability_Density_Training_1,Probability_Density_Training_2,Probability_Density_Training_3];
    Max_probability_density = max(Probability_Densities)
    if Max_probability_density == Probability_Density_Training_1
        ClassifyTrainingMLEP(i,4) = 1;
    elseif  Max_probability_density == Probability_Density_Training_2
             ClassifyTrainingMLEP(i,4) = 2;
        else
             ClassifyTrainingMLEP(i,4) = 3;
        
    end
    if Training(i,3) ~= ClassifyTrainingMLEP(i,4)
        Error_Count_Training = Error_Count_Training + 1;
    end
end
Error_Rate_Training = Error_Count_Training/numrows1;

save('ClassifyTrainingMLEP.txt' ,'ClassifyTrainingMLEP', '-ascii');

%Initializing class and error count for Testing Data
Error_Count_Testing = 0
i=0

%Using the same meand covariances and prior probabilities as those of
%Training data
for i=(1:numrows2)
    Probability_Density_Testing_1 = (Prior_Probability_1)*(exp(-0.5*((Testing(i,1:2)-Mean_1)*inv(Covariance_1)*transpose(Testing(i,1:2)-Mean_1)))/[((2*pi).^(d/2)*(det(Covariance_1)).^(1/2))])
    Probability_Density_Testing_2 = (Prior_Probability_2)*(exp(-0.5*((Testing(i,1:2)-Mean_2)*inv(Covariance_2)*transpose(Testing(i,1:2)-Mean_2)))/[((2*pi).^(d/2)*(det(Covariance_2)).^(1/2))])
    Probability_Density_Testing_3 = (Prior_Probability_3)*(exp(-0.5*((Testing(i,1:2)-Mean_3)*inv(Covariance_3)*transpose(Testing(i,1:2)-Mean_3)))/[((2*pi).^(d/2)*(det(Covariance_3)).^(1/2))])
    Probability_Densities = [Probability_Density_Testing_1,Probability_Density_Testing_2,Probability_Density_Testing_3]
    Max_probability_density = max(Probability_Densities)
    if Max_probability_density == Probability_Density_Testing_1
        ClassifyTestingMLEP(i,4) = 1
    else if  Max_probability_density == Probability_Density_Testing_2
             ClassifyTestingMLEP(i,4) = 2
        else
             ClassifyTestingMLEP(i,4) = 3
        end
    end
    if Testing(i,3) ~= ClassifyTestingMLEP(i,4)
        Error_Count_Testing = Error_Count_Testing + 1
    end  
end 
Error_Rate_Testing = Error_Count_Testing/numrows2

save('ClassifyTestingMLEP.txt' , 'ClassifyTestingMLEP' ,'-ascii');
