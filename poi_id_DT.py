#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi",'salary', 'deferral_payments', 'total_payments', 
	'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
	'total_stock_value', 'expenses', 'exercised_stock_options', 
	'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
	'to_messages','from_poi_to_this_person', 'from_messages', 
	'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#remove total from data
data_dict.pop("TOTAL", 0)

#remove travel agency in the park from data
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = dict(data_dict)

def feature_add(dict):
	for i in dict:
		if dict[i]["from_poi_to_this_person"] == "NaN":
			dict[i]["from_poi_to_this_person"] = 0
		if dict[i]["from_this_person_to_poi"] == "NaN":
			dict[i]["from_this_person_to_poi"] = 0
		if dict[i]["shared_receipt_with_poi"] == "NaN":
			dict[i]["shared_receipt_with_poi"] = 0
		poi_email_sum = sum((dict[i]["from_poi_to_this_person"], 
						     dict[i]["from_this_person_to_poi"],
						     dict[i]["shared_receipt_with_poi"]))
		dict[i]["poi_emails"] = poi_email_sum

	return dict

feature_add(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


#print features.any(min(features))
#print min(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#clf = svm.SVC(random_state = 23)
#clf = GaussianNB()



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from tester import test_classifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# 2 cross-validation methods to use
sss = StratifiedShuffleSplit(n_splits = 2, train_size = .70,
                            test_size = .30, random_state = 23)

k_fold = KFold(n_splits = 10, shuffle=True, random_state = 23)

#setting up pipeline with estimators

estimators = [("scaler", MinMaxScaler()), ("select", SelectKBest()), 
				("reduce", PCA()), ("classifier", DecisionTreeClassifier())]

#want to try without the scalar
#estimators = [("select", SelectKBest()),
#				("classifier", SVC())]

pipe = Pipeline(estimators)


score = ["recall", "precision"]

for K in score:
	'''
	#score = "recall": precision   , recall
	#score = "precision": precision  ,  recall
	params = dict(select__k = [3, 4, 5, 6, 7, 8, 9, 10, 11, "all"],
				select__score_func = [f_classif],
				#reduce__n_components = [2, 3],
				classifier__criterion = ["gini", "entropy"],
				classifier__splitter = ["best", "random"],
				classifier__max_features = ["auto", "log2", "sqrt", None],
				classifier__min_samples_split = [2, 3, 4],
				classifier__min_samples_leaf = [1, 2, 3, 4, 5],
				classifier__random_state = [23],
				classifier__class_weight = [{0:.30, 1:.70}, {0:.29, 1:.71},
			  								{0:.28, 1:.72}, {0:.27, 1:.73},
			  								{0:.24, 1:.76}, {0:.23, 1:.77} ])
	'''

	'''
	#score = "recall": accuracy .7498, precision .24646, recall .42600
	#score = "precision": accuracy .79873  ,  precision .28384, recall .33450
	params = dict(select__k = [3, 4, 5, 6, 7, 8, 9, 10, 11, "all"],
				select__score_func = [f_classif],
				#reduce__n_components = [2, 3],
				classifier__criterion = ["gini", "entropy"],
				classifier__splitter = ["best", "random"],
				classifier__max_features = ["auto", "log2", "sqrt", None],
				classifier__min_samples_split = [2, 3, 4],
				classifier__min_samples_leaf = [1, 2, 3, 4, 5],
				classifier__random_state = [23],
				classifier__class_weight = [{0:.30, 1:.70}, {0:.29, 1:.71},
			  								{0:.28, 1:.72}, {0:.27, 1:.73},
		  								{0:.24, 1:.76}, {0:.23, 1:.77} ])
	'''

	'''
	#score = "recall": accuracy .77853 , precision .25555, recall .34550 
	#score = "precision": accuracy .79953  ,  precision .28656, recall .33800
	#change one of the class weight options to {0:.265, 1:.725} 
	params = dict(select__k = [3, 4, 5, 6, 7, 8, 9, 10, 11, "all"],
				select__score_func = [f_classif],
				#reduce__n_components = [2, 3],
				classifier__criterion = ["gini", "entropy"],
				classifier__splitter = ["best", "random"],
				classifier__max_features = ["auto", "log2", "sqrt", None],
				classifier__min_samples_split = [2, 3, 4, 5],
				classifier__min_samples_leaf = [1, 2, 3],
				classifier__random_state = [23],
				classifier__class_weight = [{0:.30, 1:.70}, {0:.285, 1:.715},
			  								{0:.28, 1:.72}, {0:.265, 1:.725}])
	'''

	#score = "recall": accuracy .77853, precision .25555 , recall .3455
	#score = "precision": accuracy  .79667,  precision .27978, recall .33350 
	#change three of the class weight options. Only minor chnge in metrics 
	params = dict(select__k = [3, 4, 5, 6, 7, 8, 9, 10, 11, "all"],
				select__score_func = [f_classif],
				#reduce__n_components = [2, 3],
				classifier__criterion = ["gini", "entropy"],
				classifier__splitter = ["best", "random"],
				classifier__max_features = ["auto", "log2", "sqrt", None],
				classifier__min_samples_split = [2, 3, 4, 5],
				classifier__min_samples_leaf = [1, 2, 3],
				classifier__random_state = [23],
				classifier__class_weight = [{0:.30, 1:.70}, {0:.285, 1:.715},
			  								{0:.28, 1:.72}, {0:.275, 1:.725},
			  								{0:.27, 1:.73}, {0:.265, 1:.735},
			  								])


	my_clf = GridSearchCV(pipe, params, cv = sss, scoring = K)
	my_clf.fit(features, labels)
	clf = my_clf.best_estimator_
	print my_clf.best_params_


	
	#find and print out features
	'''
	features_k = my_clf.best_params_["select__k"]
	features_score_func = my_clf.best_params_["select__score_func"]
	select_k = SelectKBest(score_func = features_score_func, k = features_k)
	select_k.fit_transform(features, labels)
	features_selected = [features_list[1:][i] for i in select_k.get_support(indices=True)]
	print features_selected
	'''

	print test_classifier(clf, my_dataset, features_list)
	#returns features selected in best estimator

	#find and print out features
	features_k = my_clf.best_params_["select__k"]
	features_score_func = my_clf.best_params_["select__score_func"]
	select_k = SelectKBest(score_func = features_score_func, k = features_k)
	select_k.fit_transform(features, labels)
	features_selected = [features_list[1:][i] for i in select_k.get_support(indices=True)]
	print "features used are..", features_selected

	#find and print out scores
	feature_scores = select_k.scores_
	feature_scores_list = [feature_scores[i] for i in select_k.get_support(indices=True)]
	print "Scores for features used are...", feature_scores_list
	#print "Selected Features:", my_clf.best_params_["select__k"]


	'''
	#print out feature importances
	best_pipeline = my_clf.best_estimator_
	tree = best_pipeline.named_steps["classifier"]
	feature_importances = tree.feature_importances_
	feature_importances = list(feature_importances)
	feature_importances_list = [feature_importances[i] for i in select_k.get_support(indices=True)]
	print "Feature Importances for features used are...", feature_importances
	#print feature_importances_list
	


#### Long way of finding feature importances ###############################################
	#find and print out feature importances
	classifier_criterion = my_clf.best_params_["classifier__criterion"]
	classifier_splitter = my_clf.best_params_["classifier__splitter"]
	classifier_max_features = my_clf.best_params_["classifier__max_features"]
	classifier_min_samples_split = my_clf.best_params_["classifier__min_samples_split"]
	classifier_min_samples_leaf = my_clf.best_params_["classifier__min_samples_leaf"]
	classifier_random_state = my_clf.best_params_["classifier__random_state"]
	classifier_DT = DecisionTreeClassifier(criterion = classifier_criterion, 
										splitter = classifier_splitter,
										max_features = classifier_max_features,
										min_samples_split = classifier_min_samples_split,
										min_samples_leaf = classifier_min_samples_leaf,
										random_state = classifier_random_state)
	
	classifier_DT.fit(features, labels)
	feature_importances = classifier_DT.feature_importances_
	DT_importances = [feature_importances[i] for i in select_k.get_support(indices = True)]
	print "features importances for features used are...", DT_importances

	'''


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
