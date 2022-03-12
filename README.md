# Credit_Risk_Analysis

## Overview

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, you’ll oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, you’ll use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, you’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once you’re done, you’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

                                  Deliverable 1: Use Resampling Models to Predict Credit Risk
                                  
Using your knowledge of the imbalanced-learn and scikit-learn libraries, you’ll evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First, you’ll use the oversampling RandomOverSampler and SMOTE algorithms, and then you’ll use the undersampling ClusterCentroids algorithm. Using these algorithms, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

![image](https://user-images.githubusercontent.com/93686963/158003673-a0536c79-c05c-4ac5-8731-03feb16a9ca1.png)

![image](https://user-images.githubusercontent.com/93686963/158003694-439c7f38-9b4b-4392-b256-401abf642293.png)

The balanced accuracy score for this model is around 64.4%, meaning that the model predicted the credit risk accurately 64.4% of the time. This is a fairly positive score, but not great.
The precision scores for this model are very skewed toward the low-risk loans as all of the low-risk loans were correctly predicted, but nearly none of the high-risk loans were accurately predicted. This model does not work that much for identifying high-risk loan.

The recall scores for this model show that the model is better at identifying positive low-risk loans (0.68) and decent at positively identifying high-risk loans (0.61), but the recall scores are not good for either.



                              Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk 
                              
Using the SMOTEENN algorithm, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

![image](https://user-images.githubusercontent.com/93686963/158003709-7b8fdae0-aa63-4076-b28b-8f57e04d30b5.png)

![image](https://user-images.githubusercontent.com/93686963/158003721-17c7759d-c28e-4998-a2fe-51283635247c.png)

The balanced accuracy score for this model is around 63.7%, meaning that the model predicted the credit risk accurately 63.7% of the time. This is a fairly positive score, but not great.
The precision scores for this model are very skewed toward the low-risk loans as all of the low-risk loans were correctly predicted, but nearly none of the high-risk loans were accurately predicted. This model is not great for identifying high-risk loan.

The recall scores for this model show that the model is better at identifying positive low-risk loans (0.63) and decent at positively identifying high-risk loans (0.64). 


![image](https://user-images.githubusercontent.com/93686963/158003755-d70307f9-0a90-4bfd-af9c-29302932bee8.png)

![image](https://user-images.githubusercontent.com/93686963/158003789-ecd181ee-daa4-4d20-82cf-9d848b4a08ab.png)

![image](https://user-images.githubusercontent.com/93686963/158003806-11c43253-169d-4a53-b160-9d4bd825f1f3.png)

The balanced accuracy score for this model is around 63.7%, meaning that the model predicted the credit risk accurately 63.7% of the time. This is a fairly positive score, but not a  great score.
The precision scores for this model are very skewed toward the low-risk loans as all of the low-risk loans were correctly predicted, but nearly none of the high-risk loans were accurately predicted. This model is not good for identifying high-risk loan.

The recall scores for this model show that the model is better at identifying positive low-risk loans (0.44) and decent at positively identifying high-risk loans (0.59), but the recall scores are not great for either.


![image](https://user-images.githubusercontent.com/93686963/158003827-0ff8c326-c451-4c87-9f6a-74e745a8118d.png)

![image](https://user-images.githubusercontent.com/93686963/158003848-e5f34345-67f7-4e75-8c36-6d51b76aa62f.png)

![image](https://user-images.githubusercontent.com/93686963/158003861-a993d82e-a638-4e14-8a3c-465ea022ca16.png)

The balanced accuracy score for this model is around 63.7%, meaning that the model predicted the credit risk accurately 63.7% of the time. This is a fairly positive score, but not excellent.
The precision scores for this model are very skewed toward the low-risk loans as all of the low-risk loans were correctly predicted, but nearly none of the high-risk loans were accurately predicted. This model is not great for identifying high-risk loan.

The high-risk recall score for this model is fairly high at 0.70 and the low-risk recall score is average at 0.57. Compared to the previous techniques, this model is much better at identifying true high-risk loans.



![image](https://user-images.githubusercontent.com/93686963/158003894-28379fa8-63b4-4c39-abf3-23dd0e34cce1.png)

![image](https://user-images.githubusercontent.com/93686963/158003917-7ca45893-b3f2-4351-91b5-207b49d176d0.png)

![image](https://user-images.githubusercontent.com/93686963/158003980-32cf1407-8a26-43c0-a1d2-4bbb8f182239.png)

The balanced accuracy score for this model is comparatively high at 78.8%, meaning that 78.8% of the testing credit data was accurately classified.
The precision scores for this model are very skewed toward the low-risk loans as all of the low-risk loans were correctly predicted, but nearly none of the high-risk loans were accurately predicted.
The recall score for low-risk loans is very high at 0.91 and the recall score for high-risk loans is fairly high at 0.67. This shows that the classifier is good at predicting true positives for low-risk loans.

![image](https://user-images.githubusercontent.com/93686963/158004004-d228db61-8aeb-436a-b5ad-c45143c49590.png)

![image](https://user-images.githubusercontent.com/93686963/158004053-05c07225-55ed-4376-9af9-c45bc3229c3c.png)

![image](https://user-images.githubusercontent.com/93686963/158004069-1b24495c-5c85-4412-8734-464459b004bb.png)


The balanced accuracy score for this model high at around 92.5%, meaning that the classifier accurately predicted the credit risk 92.5% of the time.
The precision scores for this model are very skewed toward the low-risk loans as all of the low-risk loans were correctly predicted, but few of the high-risk loans were accurately predicted. 
The recall scores for both high and low risk were quite high in this model at 0.91 for high-risk loans and 0.94 for low-risk loans. This shows the high rate of true positives.



Summary:


All of the machine learning models had low precision scores for the high-risk loans in accurately predicting positives. The balanced accuracy score for the models varied with the lowest score for the undersampling method and a high score with the AdaBoost classifier. Recall scores also varied between models with the lowest scores for undersampling and highest with the classifying methods. Of the models created, the Easy Ensemble AdaBoost Classifier would be the best model to use to predict credit risk due to the high recall scores for both high and low risk loans, as well as an accuracy score of 92.5%. The precision for this model is still very off, indicating that the positives are not necessarily accurate, and so this model could be much improved and training and testing more data before putting it into use








