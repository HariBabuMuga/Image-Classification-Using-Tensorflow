import pandas as pd
import numpy as np 
import random as rnd
import math
# visualization
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('whitegrid')

# loading data
train_df = pd.read_csv("/home/hari/ml/train.csv")
test_df = pd.read_csv("/home/hari/ml/test.csv")
combine = [train_df, test_df]
#print(train_df.columns.values)

# preview of data
#print(train_df.head())
#print(train_df.tail())
#print(train_df.info())

#print(train_df.describe())


# analyze by pivoting features
grouping = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#print(grouping)
#print('_'*20)
grouping1 = train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#print(grouping1)
#print('_'*20)
grouping2 = train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#print(grouping2)
#print('_'*20)
grouping3 = train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#print(grouping3)


# analyze by visualizing data
g = sns.FacetGrid(train_df, col = 'Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()

# consider Pclass for model training
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

# correlating categorical features
grid = sns.FacetGrid(train_df, 'Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()

# considering banding Fare features
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()


# Wrangling data
#print("Before : ", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_new = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_new = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_new, test_new]
#print("After : ", train_new.shape, test_new.shape, combine[0].shape, combine[1].shape)


# creating new features
for dataset in combine:
	dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

n_f = pd.crosstab(train_new['Title'], test_new['Sex'])
#print(n_f)

# replacing titles
for dataset in combine:
	dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
	dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
grouping4 = train_new[['Title','Survived']].groupby(['Title'], as_index=False).mean().sort_values('Survived', ascending=False)
#print(grouping4)

# converting categorical titles to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
	dataset['Title'] = dataset['Title'].map(title_mapping)
	dataset['Title'] = dataset['Title'].fillna(0)
#print(train_new.head())

# droping unwanted features
train_new = train_new.drop(['Name', 'PassengerId'], axis=1)
test_new = test_new.drop(['Name'], axis=1)
combine = [train_new, test_new]
#print(train_new.shape, test_new.shape)

# converting categorical feature
for dataset in combine:
	dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#print(train_new.head())

grid = sns.FacetGrid(train_new, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

# Let us create Age bands and determine correlations with Survived.
train_new['AgeBand'] = pd.cut(train_new['Age'], 5)
train_new[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# Let us replace Age with ordinals based on these bands.
for dataset in combine:    
	dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
	dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
	dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
	dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
	dataset.loc[ dataset['Age'] > 64, 'Age']
#print(train_new.head())

# 
train_new = train_new.drop(['AgeBand'], axis=1)
combine = [train_new, test_new]
#print(train_new.head())

# Create new feature combining existing features
for dataset in combine:
	dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

grouping7 = train_new[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#print(grouping7)


# We can create another feature called IsAlone.
for dataset in combine:
	dataset['IsAlone'] = 0
	dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

grouping8 = train_new[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
#print(grouping8)

# Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone.
train_new = train_new.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_new = test_new.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_new, test_new]

#print(train_new.head())


# Completing a categorical feature
freq_port = train_new.Embarked.dropna().mode()[0]
#print(freq_port)

for dataset in combine:
	dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_new[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Converting categorical feature to numeric
for dataset in combine:
	dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#print(train_new.head())

# Quick completing and converting a numeric feature
test_new['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_new.head()

train_new['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_new[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# Convert the Fare feature to ordinal values based on the FareBand.
for dataset in combine:
	dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
	dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
	dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
	dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
	dataset['Fare'] = dataset['Fare'].astype(int)

train_new = train_new.drop(['FareBand'], axis=1)
combine = [train_new, test_new]
    
#print(train_new.head(10))
#print(test_new.head(10))

print(train_new.isnull().sum())
#train_new['Age'] = train_new['Age'].replace('nan',train_new['Age'].mode())
#train_new.loc[train_new['Age'] =='nan', 'Age'] = train_new['Age'].mode()
train_new["Age"].fillna(train_new['Age'].mean(), inplace=True)
print(train_new.isnull().sum())
# Model, predict and solve
x_train = np.array(train_new.drop('Survived', axis=1))
Y = np.array(train_new["Survived"])
x_test = np.array(test_new.drop('PassengerId', axis=1).copy())
#print(x_train.shape, Y.shape, x_test.shape)

#print(type(x_train))
#print(type(Y))
#print(type(x_test))

x0 = np.ones(len(Y))
#print(x0)
X = np.column_stack((x0,x_train))
B = np.array([0.001,0.003,0.02,-0.05,0.009,0.02,0.09,0.04])
#print(x_train.shape, X.shape, B.shape, Y.shape)

alpha = 0.01

def sigmoid(sof):
	equation = 1/(1+np.exp(-sof))
	return(equation)

# defining cost function 
def cost_function(X,Y,B):
	sof = X.dot(B)
	lr = sigmoid(sof)
	LR = np.nan_to_num(lr)
	cost =  np.sum(-(Y.dot(np.log(LR))+(1-Y).dot(np.log(1-LR)))/(Y.shape[0]))
	return(cost)
#print(cost_function(X,Y,B))

# defining gradient ascent
def gradient_ascent(X,Y,B,alpha,iterations):    #batch_gradient_descent
	cost_history=[]
	x_range = [0]*iterations
	for i in range(iterations):
		sof = X.dot(B)
		lr = sigmoid(sof)
		LR = np.nan_to_num(lr)
		loss = Y - sof
		gradient = np.nan_to_num(X.T.dot(loss)/(Y.shape[0]))
		B = B + alpha*gradient
		total_cost  = cost_function(X,Y,B)  # total cost with updated weights
		cost_history.append(total_cost)
		#print(i)
		x_range[i] = i 
	return(B,cost_history,x_range)
B_new, total_cost,x_range = gradient_ascent(X,Y,B,alpha,100000)

print(B_new)
y_pred = np.nan_to_num(X.dot(B_new))
print(y_pred)
plt.plot(x_range,total_cost,color='red')
plt.show()



