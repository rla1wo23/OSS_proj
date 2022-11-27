#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/rla1wo23/OSS_proj.git
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def load_dataset(dataset_path):
	data_df=pd.read_csv(dataset_path)
	return data_df
	#To-Do: Implement this function

def dataset_stat(dataset_df):	
	n_rows,n_feats = dataset_df.shape
	n_feats=n_feats-1
	dataset_df.target.shape
	rank_zero=len(dataset_df.loc[dataset_df['target']==0])
	rank_one=len(dataset_df.loc[dataset_df['target']==1])
	return n_feats, rank_zero, rank_one
	#To-Do: Implement this function

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	data=dataset_df.drop(columns="target", axis=1)
	target=dataset_df['target']
	x_train, x_test, y_train, y_test=train_test_split(data,target,test_size=testset_size,random_state=2)
	return x_train, x_test, y_train, y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	pipe=make_pipeline(StandardScaler(),RandomForestClassifier())
	pipe.fit(x_test,y_test)

	dt_cls=DecisionTreeClassifier()
	dt_cls.fit(x_train,y_train)

	acc=accuracy_score(y_test,dt_cls.predict(x_test))
	pre=precision_score(dt_cls.predict(x_test),y_test)
	recall=recall_score(y_test,dt_cls.predict(x_test))
	return acc, pre, recall

	#To-Do: Implement this function

def random_forest_train_test(x_train, x_test, y_train, y_test):
	rf_cls=RandomForestClassifier()
	rf_cls.fit(x_train,y_train)

	acc=accuracy_score(y_test,rf_cls.predict(x_test))
	pre=precision_score(rf_cls.predict(x_test),y_test)
	recall=recall_score(y_test,rf_cls.predict(x_test))
	return acc, pre, recall
	#To-Do: Implement this function

def svm_train_test(x_train, x_test, y_train, y_test):
	svm_cls=SVC()
	svm_cls.fit(x_train,y_train)

	acc=accuracy_score(y_test,svm_cls.predict(x_test))
	pre=precision_score(svm_cls.predict(x_test),y_test)
	recall=recall_score(y_test,svm_cls.predict(x_test))
	return acc, pre, recall

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)