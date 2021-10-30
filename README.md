# Principal-Component-Analysis

PCA is utilised in exploratory data analysis as well as the creation of prediction models. It is widely used to reduce dimensionality by projecting each data point onto only the first few main components to produce lower-dimensional data while retaining as much of the data's variance as feasible. The first principal component is defined as the direction that minimises the variance of the predicted data.

Obviously, reducing the number of variables in a data collection reduces accuracy, but the idea in dimensionality reduction is to trade a little accuracy for simplicity. Because smaller data sets are easier to study and display, and because machine learning algorithms can analyse data much more easily and quickly without having to deal with superfluous factors.

To summarise, the goal behind PCA is straightforward: decrease the number of variables in a data set while retaining as much information as feasible.

## CIFAR_10

CIFAR_10 is a dataset taken from importing keras api

Dicitionary is made and model trained using labels

Using sklearn.decomposition, pca is imported and model is made specifying the components in the tabular form

Defining model and visualization of the 2 components of pca is plotted 

Model is compiled, fitted and trained, after which testing accuracy is considered

## Cervical_Cancer_Risk_Classification

In this project, cancer risk is classified on the basis of various factors such as age, STD's,  biopsy

Data cleaning is done and data preprocessing and decomposition using PCA

Plotting done between principle component and eigenvalue. KMeans cluster is used in this 

Dataset : kag_risk_factors_cervical_cancer.csv ( https://github.com/Akshatpattiwar512/Principal-Component-Analysis/blob/main/kag_risk_factors_cervical_cancer.csv )

## Classifier_Boundaries_Visualization_Using_PCA(Kernel)

In this project, different classifiers vs kernel pca is plotted

Also the Component 1 and 2 of boundary line test( Training and testing) data is plotted with different types of pca(ply, sigmoid, cosine, rbf, linear)

Dataset link : https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/version/2

## Credit_Card_Dataset_for_Clustering_using_PCA

This project is about clustering modeling using KMeans, interpretation of clusters, pca to transform 2 dimensions visualization

Different types of clusters plotted on credit card usage behaviour( 6 different types )

Dataset link : https://www.kaggle.com/arjunbhasin2013/ccdata

## Fruit_Classification_PCA,_SVM,_KNN,_Decision_Tree

Data cleaning, preprocessing and different classification of dataset
n-gram tfidf vectorizer, Random Forest Classifier is used. Also NLP, bagging and boosting is compared. Loss and accuracy graph is plot.


Dataset link : https://www.kaggle.com/datafiniti/grammar-and-online-product-reviews

## Kannada_MNIST_Logistic,_DT,_PCA,_XGBOOST

Splitting training and testing data and visualization using labels and 

Model trained using the following ml models and accuracy percent calculated

1.Logistic Regression

2.Decision Tree

3.PCA svm

4.XGBOOST

5.AdaBoost

Barplot comparison is done between algorithms and accuracy score

Dataset : Kannada-MNIST

## Variational_Autoencoder_with_PyTorch_vs_PCA

PCA with 3 dimensions and pytorch encoder : Data loader building 

Embedding done after evaluation model and 3d plot embedding of PCA and VAE 

Dataset link : https://www.kaggle.com/brynja/wineuci

## For google colab(Importing of dataset from kaggle)

! pip install kaggle

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download < Dataset-name >

! unzip < zip-file >
