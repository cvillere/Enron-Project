#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi", "from_this_person_to_poi", "from_poi_to_this_person",
					"to_messages", "from_messages", "shared_receipt_with_poi"] # You will need to use more features

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
sss = StratifiedShuffleSplit(n_splits = 2, train_size = .60,
                            test_size = .40, random_state = 23)

k_fold = KFold(n_splits = 10, shuffle=True, random_state = 23)

#setting up pipeline with estimators

estimators = [("scaler", MinMaxScaler()), ("select", SelectKBest()), 
				("reduce", PCA()), ("classifier", GaussianNB())]


pipe = Pipeline(estimators)


score = ["precision", "recall", "f1"]

for K in score:

	'''
	#score = ["precision"]: accuracy .84207, precision  .31715, recall .1600
	#score = ["recall"]: accuracy  .84613, precision  .37209 , recall .22400
	#score = ["f1"]: accuracy   .84613,  precision   0.37209, recall .22400
	params = dict(select__k = [1, 2, 3, 4, 5, "all"],
				select__score_func = [f_classif, chi2])
				#reduce__n_components = [1])
	'''

	#took out a bunch of features from features_list. Left features
	#bonus, total_stock_value, excercised_stock_options, salary, total_payments
	#and poi_emails
	#score = ["precision"]: accuracy .85147, precision  .40484, recall .24250 
	#score = ["recall"]: accuracy  .8578, precision  .44589 , recall .27400
	#score = ["f1"]: accuracy   .8578,  precision  .44589 , recall .27400
	params = dict(select__k = [1, 2, 3, 4, 5, "all"],
				select__score_func = [f_classif, chi2])
				#reduce__n_components = [1])			


	'''
	#added back shared_receipt_with_poi
	#score = ["precision"]: accuracy .8250, precision .30897 , recall .25150 
	#score = ["recall"]: accuracy  .85087, precision  .39811 , recall .23150
	#score = ["f1"]: accuracy  .8252 ,  precision .30897 , recall .25150
	params = dict(select__k = [1, 2, 3, 4, 5, "all"],
				select__score_func = [f_classif, chi2])

	'''

	'''
	#took away finacial features, and just used email features
	#score = ["precision"]: accuracy .80267, precision  0.01621, recall 0.1300 
	#score = ["recall"]: no numbers
	#score = ["f1"]: accuracy  .84222,  precision .00237 , recall 0.00100 
	params = dict(select__k = [1, 2, 3, 4, 5, "all"],
				select__score_func = [f_classif, chi2])
				#reduce__n_components = [1])
	'''

	my_clf = GridSearchCV(pipe, params, cv = sss, scoring = K)
	my_clf.fit(features, labels)
	clf = my_clf.best_estimator_
	print my_clf.best_params_


	print test_classifier(clf, my_dataset, features_list)
	#returns features selected in best estimator

	#find and print out features
	features_k = my_clf.best_params_["select__k"]
	#features_score_func = my_clf.best_params_["select__score_func"]
	select_k = SelectKBest(score_func = f_classif, k = features_k)
	select_k.fit_transform(features, labels)
	features_selected = [features_list[1:][i] for i in select_k.get_support(indices=True)]
	print "features used are..", features_selected

	#find and print out scores
	feature_scores = select_k.scores_
	feature_scores_list = [feature_scores[i] for i in select_k.get_support(indices=True)]
	print "Scores for features used are...", feature_scores_list




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
