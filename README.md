## **WINE QUALITY PREDICTION** ##
 **I love everything that’s old — old friends, old times, old manners, old books, old wine. — Oliver Goldsmith**
 
 ![](https://miro.medium.com/max/1050/1*2ayKmvVZCYaLPl-nmLLp5g.png)
 
## **INTRODUCTION** ##

Most of us love wine. A good wine always makes the occasion better. Its always the quality of the wine that matters . Lets apply machine learning model to figure  out what makes a qood quality wine.
For this project , I have used the Wine Dataset from UC Irvine Machine Learning Repository.It consists 11 input variables to predict the quality of wine .
**1.Fixed acidity

2.Volatile acidity

3.Citric acid

4.Residual sugar

5.Chlorides

6.Free sulfur dioxide

7.Total sulfur dioxide

8.Density

9.pH

10.Sulfates

11.Alcohol**

We can predict the quality of wine using different models which yeilds different results. Here we use **Scikit-learn’s Decision Tree Classifier**.Decision trees are intuitive and easy to build . Let's learn more about Decision trees.

# **Decision Tree** #
![Decision tree](https://miro.medium.com/max/945/1*f_tt4OIzuY4yoPrnFP0gdA.png)

Decision trees are a popular model, used in operations research, strategic planning and machine learning. Each square above is called a node, and the more nodes you have, the more accurate your decision tree will be (generally). The last nodes of the decision tree, where a decision is made, are called the leaves of the tree.

The root node (the first decision node) partitions the data based on the most influential feature partitioning. There are 2 measures for this, Gini Impurity and Entropy.

**Entropy**

The root node (the first decision node) partitions the data using the feature that provides the most information gain.
Information gain tells us how important a given attribute of the feature vectors is.
It is calculated as:

Information Gain=entropy(parent)–[average entropy(children)]
Where entropy is a common measure of target class impurity, given as:

Entropy=Σi–pilog2pi
where i is each of the target classes.

**Gini Impurity**

Gini Impurity is another measure of impurity and is calculated as follows:

Gini=1–Σip2i

Gini impurity is computationally faster as it doesn’t require calculating logarithmic functions, though in reality which of the two methods is used rarely makes too much of a difference.

![Classification tree of red wine before pruning](https://cdn-images-1.medium.com/max/800/0*zT39Hb7MzF5rMt7b)



**Now let's predict the quality of wine**

## Importing libraries ##

First let's import all the required libraries

```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import classification_report
```

numpy will be used for making the mathematical calculations more accurate, pandas will be used to work with file formats like csv, xls etc. and sklearn (scikit-learn) will be used to import our classifier for prediction. Seaborn provides a high-level interface for drawing attractive and informative statistical graphics.It is a Python data visualization library based on matplotlib.
Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy.

from sklearn.model_selection import train_test_split is used to split our dataset into training and testing data, more of which will be covered later. The next import, from sklearn import preprocessing is used to preprocess the data before fitting into predictor, or converting it to a range of (-1,1), which is easy to understand for the machine learning algorithms. Next import, from sklearn import tree is used to import our decision tree classifier, which we will be using for prediction.
A Classification report is used to measure the quality of predictions .

## **Reading data** ##

The very next step is importing the data that  we will be using. For this project, we will be using the Wine Dataset from UC Irvine Machine Learning Repository.
```
data=pd.read_csv('winedataset.csv')
```

We use pd.read_csv() function in pandas to import the data by giving the name of the dataset . We use ‘;’ (semi-colon) as  separator to obtain the csv in a more structured format.

Now we have to analyse the dataset. First we will see what is inside the data set by seeing the first five values of dataset by head() command.
```
data.head()
```

```
  fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
0            7.4              0.70         0.00             1.9      0.076   
1            7.8              0.88         0.00             2.6      0.098   
2            7.8              0.76         0.04             2.3      0.092   
3           11.2              0.28         0.56             1.9      0.075   
4            7.4              0.70         0.00             1.9      0.076   

   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
0                 11.0                  34.0   0.9978  3.51       0.56   
1                 25.0                  67.0   0.9968  3.20       0.68   
2                 15.0                  54.0   0.9970  3.26       0.65   
3                 17.0                  60.0   0.9980  3.16       0.58   
4                 11.0                  34.0   0.9978  3.51       0.56   

   alcohol  quality  
0      9.4        5  
1      9.8        5  
2      9.8        5  
3      9.8        6  
4      9.4        5 
```
head() Information Of Wine Dataset

## **Analysing the data**  ##
Now we need to analyse the data . First lets see the distribution of quality variable. Lets make the histogram of quality variable.

```
x=data.quality
fig=plt.hist(x,bins=10)
```
![Quality histogram](https://miro.medium.com/max/1050/1*mby2pZfP2mnyi5wP_Uh9sQ.png)


Next lets see the correlation between variables. This will help us  to understand the relationship between variables.

```
corr = data.corr()
matplotlib.pyplot.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
```
![correlation graph](https://miro.medium.com/max/1050/1*dhDaVItzF4dzxLlJ8Kyhlw.png)

We can observe that there are certain variables that  are strongly correlated to the *quality*. Its most likely that these features are very important for determining the quality.

## **Convert to a Classification Problem**  ##

Every machine learning model has two things **Features** and **Labels** . Features are the part of a dataset which are used to predict the label. Labels  on the other hand are mapped to features. After the model has been trained, we give features to it, so that it can predict the labels.

So in this dataset, we need to find the quality of the wine. So quality is our label and and all other columns are features. We need to separate features and label into two different data frames.

```
y=data.quality
X=data.drop('quality',axis=1)
```
Here we stored *quality* in y and all other attributes in X.

## **Split data**  ##

Next we need to split our dataset into train and test data sets . Train data set is used to train the model and test data set is used to test the created model.This splitting is done by train_test_split() function.

```
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
```
_Splitting Into Test And Train Data_


We have used test_size=0.1 which means 10% of the original data is used for testing and remaining 90% of data is used for training the model.
Now lets print the first five elements of data we have split using head() function.

```
X_train.head()
```

```
      fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
591             6.6              0.39         0.49             1.7      0.070   
1196            7.9              0.58         0.23             2.3      0.076   
1128           10.0              0.43         0.33             2.7      0.095   
640             9.9              0.54         0.45             2.3      0.071   
389             9.6              0.38         0.31             2.5      0.096   

      free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
591                  23.0                 149.0  0.99220  3.12       0.50   
1196                 23.0                  94.0  0.99686  3.21       0.58   
1128                 28.0                  89.0  0.99840  3.22       0.68   
640                  16.0                  40.0  0.99910  3.39       0.62   
389                  16.0                  49.0  0.99820  3.19       0.70   

      alcohol  
591      11.5  
1196      9.5  
1128     10.0  
640       9.4  
389      10.0  
```
_Training Data Using head()_


## **Data normalization**  ##

We have now obtained the data we want to use. Once we have got the data we need to normailze the data. Normalization is the pre-processing in which data is converted to fit in a range of -1 and 1.

```
X_train_scaled = preprocessing.scale(X_train)
print X_train_scaled
```
_Train Data Preprocessing_


After pre-processing
```
array([[-0.87446643,  0.79860489, -0.6229427 , ...,  0.70432307,
         0.06387655,  0.82832417],
       [-1.10298858,  0.79860489, -0.98056848, ...,  1.15526353,
        -0.98197072,  0.92191388],
       [ 1.41075502,  2.0455708 ,  0.14339824, ..., -2.06573976,
         3.02711047, -0.20116258],
       ...,
       [-0.64594429,  0.28848248,  1.36954375, ...,  0.12454248,
        -0.51714971, -0.10757288],
       [ 1.86779932, -0.67508208,  1.36954375, ..., -1.87247956,
         0.58680018, -0.4819317 ],
       [ 0.61092752, -0.73176235,  0.19448764, ..., -1.55037923,
        -0.16853395,  0.07960653]])
```

We can observe that all the data are within the range -1 to 1.

## **Modelling** ##

Now lets train our algorithm so that it can predict the quality of wine. We can do it by importing DecisionTreeClassifier() and we need to use fit() to train it.

```
classifier=tree.DecisionTreeClassifier()
classifier.fit(X_train,y_train)

```
Now lets check the efficiency of algorithm in predicting the quality of wine. We use score() function to check the accuracy.

```
efficiency=classifier.score(X_test,y_test)
print('Efficiency score',efficiency)
```

```
Efficiency score 0.69375
```
As we we can see the efficiency of our algorithm in finding the quality is 0.69375
This score can change over time depending on the size of your dataset and shuffling of data when we divide the data into test and train, but you can always expect a range of ±5 around your first result.


Now lets predict the data. We have trained our classifier with features, so let's obtain the labels using predict() function.

```
prediction=classifier.predict(X_test)
prediction
```

```
array([6, 6, 5, 6, 5, 6, 5, 5, 6, 6, 8, 5, 5, 6, 6, 5, 7, 5, 6, 7, 6, 5,
       6, 6, 6, 5, 6, 5, 6, 5, 5, 7, 5, 5, 7, 6, 5, 6, 6, 7, 6, 5, 6, 6,
       7, 6, 5, 5, 6, 5, 6, 7, 7, 5, 5, 5, 6, 5, 6, 6, 5, 7, 5, 7, 5, 6,
       6, 5, 6, 7, 5, 5, 6, 6, 5, 5, 5, 6, 5, 7, 6, 6, 5, 6, 7, 5, 5, 5,
       7, 6, 6, 6, 6, 6, 5, 6, 5, 7, 5, 4, 7, 6, 6, 4, 6, 6, 5, 6, 6, 5,
       5, 5, 5, 6, 5, 6, 5, 5, 6, 5, 5, 7, 6, 7, 7, 6, 5, 5, 5, 5, 4, 5,
       6, 5, 6, 5, 6, 7, 5, 6, 5, 6, 7, 5, 7, 6, 6, 5, 6, 7, 7, 6, 5, 6,
       5, 4, 6, 7, 5, 4], dtype=int64

```
Our predicted information is in prediction but it has many columns to comapre with the expected labels we stored in y_test.So we need to take first five entries of prediction and compare it with y_test.

```
x=np.array(prediction).tolist()
print('The prediction:\n')
for i in range (0,5):
    print(x[i])
print('The expectation:\n')
y_test.head()
```
_Comparing The Predicted And Expected Labels_


We converted the numpy array into list so that we can compare easily.Then we printed the first five elements of that list using for loop. Now we print the five values that we were expecting, which were stored in y_test using head() function. The output looks something like this

```
The prediction:

6
5
5
5
5

The expectation:

Out[121]:
3       6
1262    5
1391    5
1222    6
758     5
Name: quality, dtype: int64
```
_The Output_


We can notice that almost all of the values in the prediction are similar to the expectations. Our predictor got wrong just once, predicting 4th entry 6 as 5 but that’s it. This gives us the accuracy of 80% for 5 examples. Of course, as the examples increases the accuracy goes down, precisely to 0.621875 or 62.1875%, but overall our predictor performs quite well, in-fact any accuracy % greater than 50% is considered as great.

We can also print the classification report to the above problem by importing classification_report from sklearn.metrics and can find the average accuracy .

```
print(classification_report(y_test,prediction))
```

```
        precision    recall  f1-score   support

           3       0.00      0.00      0.00         0
           4       0.00      0.00      0.00         2
           5       0.81      0.70      0.75        80
           6       0.59      0.73      0.66        56
           7       0.70      0.64      0.67        22

    accuracy                           0.69       160
   macro avg       0.42      0.41      0.41       160
weighted avg       0.71      0.69      0.70       160

```
_Classification report_

**Thus we predict the quality of wine using Decison tree model.**

## **Thank you!!**  ##

Assignment during Online internship with DLithe(www.dlithe.com)
