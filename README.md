<H3>SASINTHAR P</H3>
<H3>212223230199</H3>
<H3>EX. NO.1</H3>
<H3>26/08/2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

#  PROGRAM:
```
## IMPORT LIBRARIES

from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

## READ THE DATASET

pythondf=pd.read_csv("Churn_Modelling.csv")

## CHECKING THE DATA
df.head()
df.tail()
df.columns

## Check the missing data
df.isnull().sum()

## Check for Duplicates

df.duplicated()

## Assigning Y

y = df.iloc[:, -1].values
print(y)

## Check for outliers
df.describe()

## Dropping string values data from dataset

data = df.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

## Normalize the dataset
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

## Split the dataset
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)

## Training and testing model

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

## OUTPUT:
## DATA CHECKING

<img width="812" height="147" alt="image" src="https://github.com/user-attachments/assets/c77d049b-eb62-404e-934e-8f536ca3b52e" />

## Check the missing data

<img width="340" height="686" alt="image" src="https://github.com/user-attachments/assets/ad4b1add-8144-4c32-9aec-ca319655ad1c" />

## Check for Duplicates

<img width="328" height="626" alt="image" src="https://github.com/user-attachments/assets/5b8bc197-21e1-483f-bd1f-d827cf00d6f1" />

## Assigning Y

<img width="272" height="75" alt="image" src="https://github.com/user-attachments/assets/0ef2a7fe-042d-4065-aff7-2e9087e9be2d" />

## OUTLIERS

<img width="1452" height="396" alt="image" src="https://github.com/user-attachments/assets/6790e06e-1b25-41ba-b25d-4aca0f290222" />

## Checking datasets after dropping string values data from dataset:

<img width="1405" height="276" alt="image" src="https://github.com/user-attachments/assets/de480bc0-d4ca-4f34-9b9d-67006f3598f1" />

## Normalize the dataset

<img width="837" height="620" alt="image" src="https://github.com/user-attachments/assets/1e03c04d-0755-4b57-995f-aa8f91d1ee4c" />

## Split the dataset

<img width="514" height="198" alt="image" src="https://github.com/user-attachments/assets/6242f0b8-8986-49fb-8ceb-91c7bf06298a" />

## Training and testing model

<img width="556" height="527" alt="image" src="https://github.com/user-attachments/assets/bed42db6-9a5b-4b9a-91aa-8c028ca0229d" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.

