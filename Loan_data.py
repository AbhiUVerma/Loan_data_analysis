#importing libraries...

from io import StringIO
import requests
import json
import pandas as pd


# Importing a dataset...
loan_data = pd.read_csv('E:\GIT\Loan_data_analysis\Loan_prediction_data.csv', dtype={'id':str})
loan_data.head()

# importing libraries...
import numpy as np
import matplotlib.pyplot as plt

#displaying first row of loan dataset...
loan_data.iloc[0]

#displaying summary of dataset...
loan_data.describe()

# Checking & printing count of null values
null_counts = loan_data.isnull().sum()
null_counts
loan_data = loan_data.dropna(axis=0)

#displaying dimension of dataset...
loan_data.shape

#return the count of all the categorical values...
loan_data['loan_status'].value_counts()

#displaying graph of loan status...
loan_data['loan_status'].value_counts().plot(kind= 'barh', color = 'orange', title = 'Possible Loan Status', alpha = 0.75)
plt.show()

#considering only two categorical options: Fully paid & Charged off...
loan_data = loan_data[(loan_data['loan_status'] == "Fully Paid") | (loan_data['loan_status'] == "Charged Off")]

#displaying graph of Fully paid & Charged off...  
loan_data['loan_status'].value_counts().plot(kind= 'barh', color = 'blue', title = 'Simplified Possible Loan Status', alpha = 0.55)
plt.show()

#converting categorical values into numerical by taking Fully paid : 1 & Charged off : 0...
status_replace = {
    "loan_status" : {
        "Fully Paid": 1,

        
        
        "Charged Off": 0,
    }
}
loan_data = loan_data.replace(status_replace)

#displaying count of both categorical values...
loan_data['loan_status'].value_counts()

# Removing unused columns
loan_data = loan_data.drop(['issue_d','zip_code'], axis =1)
loan_data = loan_data.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 
                            'grade', 'sub_grade', 'emp_title'], axis =1)
loan_data = loan_data.drop(['recoveries', 'collection_recovery_fee', 
                              'last_pymnt_d', 'last_pymnt_amnt'], axis =1)

loan_data = loan_data.drop(['out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
                              'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee'], axis =1)

# displaying dimension of dataset after deleting unused columns...
loan_data.shape

# deleting columns with null values...  
orig_columns = loan_data.columns
drop_columns = []
for col in orig_columns:
    col_series = loan_data[col].dropna().unique()
    if len(col_series) == 1:
        drop_columns.append(col)
loan_data = loan_data.drop(drop_columns, axis = 1)
drop_columns

# displaying dimensions after deleting unsused and null col... 
loan_data.shape

# displaying data types of columns...
print(loan_data.dtypes.value_counts())

# displaying data of row, count start with 0...
object_columns_df = loan_data.select_dtypes(include=["object"])
print(object_columns_df.iloc[0])

# replacing categorical value into numerical value...
status_replace_term = {
    "term" : {
        " 36 months": 0,
        " 60 months": 1,
    }
}
loan_data = loan_data.replace(status_replace_term)

#displayong count of both categorical values...
loan_data['term'].value_counts()


# replacing categorical value into numerical value...
status_replace_verification_status = {
    "verification_status" : {
        "Verified": 1,
        "Source Verified": 2,
        "Not Verified" : 3,
    }
}
loan_data = loan_data.replace(status_replace_verification_status)

#displayong count of both categorical values...
loan_data['verification_status'].value_counts()

# replacing categorical value into numerical value...
Emp_length_data = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}

loan_data = loan_data.replace(Emp_length_data)

#displayong count of both categorical values...
loan_data['emp_length'].value_counts()

#creating dataset for knn predictive analysis...
Final_loan_data_KNN = pd.DataFrame(loan_data, columns = ['loan_amnt','installment','term',
                                                         'int_rate','emp_length','annual_inc' ,'loan_status'])
    
#Displaying head() for knn dataset...  
Final_loan_data_KNN.head()    

#creating file for Knn dataset file...
Final_loan_data_KNN.to_csv('Final_loan_data_KNN.csv' , sep=',')
new_df_file=pd.read_csv('Final_loan_data_KNN.csv',index_col=0)
print(new_df_file.head())

# load libraries
import sys
from numpy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import neighbors, tree, naive_bayes
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#displaying histogram for loan amount...
Final_loan_data_KNN['loan_amnt'].plot(kind="hist", bins=10)
#it shows data is right skewed(positive) which means the mean is greater than median in this column...

#displaying bar plot for the employement length column...
Final_loan_data_KNN['emp_length'].value_counts().plot(kind='bar')

#displaying bar plot for loan status data...
Final_loan_data_KNN['loan_status'].value_counts().plot(kind='bar')

#creating a data frame for independent variable...    
X = Final_loan_data_KNN.drop("loan_status", axis=1, inplace = False)
X.head()

#creating a data frame for dependent variable which is loan status... 
y = Final_loan_data_KNN.loan_status
y.head()


#normalizing the independent data... 
from sklearn.preprocessing import MinMaxScaler

Scaler = MinMaxScaler()

X[['loan_amnt','installment','term', 'int_rate',
   'emp_length','annual_inc' ]] = Scaler.fit_transform(X[[
               'loan_amnt','installment','term',
           'int_rate','emp_length','annual_inc' ]])

X.head()

#testing and training data... 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, 
                                                    random_state=123)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



# Building the k Nearest Neighbor Classifier 

# start out with the number of classes for neighbors
data_knn = KNeighborsClassifier(n_neighbors = 2, metric='euclidean')
data_knn

from sklearn.preprocessing import MinMaxScaler

data_knn.fit(x_train, y_train)

data_knn.predict(x_test)

len(data_knn.predict(x_test))


# R-square from training and test data
rsquared_train = data_knn.score(x_train, y_train)
rsquared_test = data_knn.score(x_test, y_test)
print ('Training data R-squared:')
print(rsquared_train)
print ('Test data R-squared:')
print(rsquared_test)

# confusion matrix

from sklearn.metrics import confusion_matrix

knn_confusion_matrix = confusion_matrix(y_true = y_test, y_pred = data_knn.predict(x_test))
print("The Confusion matrix:\n", knn_confusion_matrix)

# visualize the confusion matrix
plt.matshow(knn_confusion_matrix, cmap = plt.cm.Blues)
plt.title("KNN Confusion Matrix\n")
plt.ylabel('True label')
plt.xlabel('Predicted label')
for y in range(knn_confusion_matrix.shape[0]):
    for x in range(knn_confusion_matrix.shape[1]):
        plt.text(x, y, '{}'.format(knn_confusion_matrix[y, x]),
                horizontalalignment = 'center',
                verticalalignment = 'center',)
plt.show()


#Generate the classification report
from sklearn.metrics import classification_report
knn_classify_report = classification_report(y_true = y_test, 
                                           y_pred = data_knn.predict(x_test))
print(knn_classify_report)

