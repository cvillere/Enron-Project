#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi",'salary','total_payments', 
        'bonus','total_stock_value', 'exercised_stock_options'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#task 1.5: Data Exploration: See Rubric

#function to determine total number of data points

'''
def total_entries(input):
    sum = 0
    for i in input:
            sum = sum + 1
    return sum

print "total entries:", total_entries(data_dict)

#function to determine number of POIs

def poi_true(input):
    sum = 0
    for i in input:
        if input[i]['poi'] == 1.0:
            sum = sum + 1
    return sum

print "total_POIs:", poi_true(data_dict)


#there are features with missing values. For example, total payments 
def entries_total_NANs(input):
    sum = 0
    for i in input:
        if input[i]['total_payments'] == "NaN":
            sum = sum + 1
    return sum

print "NaN_TPs:", entries_total_NANs(data_dict)

#Or total stock value

def entries_total_NANs(input):
    sum = 0
    for i in input:
        if input[i]['total_stock_value'] == "NaN":
            sum = sum + 1
    return sum

print "NaN_TSVs:", entries_total_NANs(data_dict)

#or exercised stock options

def entries_total_NANs(input):
    sum = 0
    for i in input:
        if input[i]['exercised_stock_options'] == "NaN":
            sum = sum + 1
    return sum

print "NaN_ESOs:", entries_total_NANs(data_dict)

'''
### Task 2: Remove outliers
#removes TOTAL entry from dataset
data_dict.pop("TOTAL", 0)


#removes TRAVEL AGENCY IN THE PARK
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)



###Investigate outliers by plotting several variables
###against salary
'''
data = featureFormat(data_dict, features)
features = ["salary", "bonus"]
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



features = ["salary", "total_payments"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    total_payments = point[1]
    matplotlib.pyplot.scatter( salary, total_payments )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("total_payments")
matplotlib.pyplot.show()


features = ["salary", "total_stock_value"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    total_stock_value = point[1]
    matplotlib.pyplot.scatter( salary, total_stock_value )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("total_stock_value")
matplotlib.pyplot.show()


features = ["salary", "from_this_person_to_poi"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    from_this_person_to_poi = point[1]
    matplotlib.pyplot.scatter( salary, from_this_person_to_poi )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("from_this_person_to_poi")
matplotlib.pyplot.show()


features = ["salary", "exercised_stock_options"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    exercised_stock_options = point[1]
    matplotlib.pyplot.scatter( salary, exercised_stock_options )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("exercised_stock_options")
matplotlib.pyplot.show()

#No additional entries were removed due to being 
#determined they were outliers
'''

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

#feature I created is not used in final algorithm.
#feature_add(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


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

estimators = [("scalar", MinMaxScaler()), ("select", SelectKBest()), 
				("reduce", PCA()), ("classifier", SVC())]

#want to try without the scalar
#estimators = [("select", SelectKBest()),
#				("classifier", SVC())]

pipe = Pipeline(estimators)


score = ["f1"]

for K in score:
	

	##below reaches the Precision and Recall requirement for project.
	#score = ["f1"]: accuracy .85060, precision .42191, recall .32550.
	#algorithm run when my feature is NOT in the feature list
	params = dict(select__k = [1, 2, 3, 4],
				select__score_func = [f_classif, chi2],
				#reduce__n_components = [1, 2],
			  	classifier__C = [1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27],
			  	classifier__gamma = [1.60, 1.65, 1.68, 1.69, 1.70, 1.72, 1.74],
			  	classifier__kernel = ["rbf"],
			  	classifier__class_weight = [{0:.20, 1:.80}, {0:.15, 1:.85},
			  								{0:.19, 1:.81}, {0:.14, 1:.86},
			  								{0:.18, 1:.82}, {0:.13, 1:.87},
			  								{0:.17, 1:.83}, {0:.12, 1:.88},
			  								{0:.16, 1:.84}, {0:.11, 1:.89}])
	

	
	'''
	##below reaches the Precision and Recall requirement for project.
	#score = ["f1"]: accuracy .85060, precision .42191, recall .32550.
	#algorithm run when my feature is in the feature list
	params = dict(select__k = [1, 2, 3, 4],
				select__score_func = [f_classif, chi2],
				#reduce__n_components = [1, 2],
			  	classifier__C = [1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27],
			  	classifier__gamma = [1.60, 1.65, 1.68, 1.69, 1.70, 1.72, 1.74],
			  	classifier__kernel = ["rbf"],
			  	classifier__class_weight = [{0:.20, 1:.80}, {0:.15, 1:.85},
			  								{0:.19, 1:.81}, {0:.14, 1:.86},
			  								{0:.18, 1:.82}, {0:.13, 1:.87},
			  								{0:.17, 1:.83}, {0:.12, 1:.88},
			  								{0:.16, 1:.84}, {0:.11, 1:.89}])

	'''
	'''

	#score = ["f1"]: accuracy .82320, precision .28748, recall .22050
	#algorithm run when my feature is NOT in the feature list
	params = dict(select__k = [1, 2, 3, 4],
				select__score_func = [f_classif, chi2],
				#reduce__n_components = [1],
			  	classifier__C = [0.89, 0.90],
			  	classifier__gamma = [1.425, 1.43, 1.435],
			  	classifier__kernel = ["rbf"],
			  	classifier__class_weight = [{0:.22, 1:.78}, {0:.24, 1:.76},
			  								{0:.18, 1:.82}, {0:.14, 1:.86},
			  								{0:.179, 1:.819}, {0:.115, 1:.885},
			  								])


	'''

	'''
	#score = ["f1"]:accuracy .76,  precision .26649,   recall .45650 
	#algorithm run when my feature is NOT in the feature list
	params = dict(select__k = [1, 2, 3, 4],
				select__score_func = [f_classif, chi2],
				#reduce__n_components = [1, 2],
			  	classifier__C = [.95, .97],
			  	classifier__gamma = [1.41, 1.425, 1.43, 1.435],
			  	classifier__kernel = ["rbf"],
			  	classifier__class_weight = [{0:.20, 1:.80}, {0:.15, 1:.85},
			  								{0:.189, 1:.809}, {0:.14, 1:.86},
			  								{0:.179, 1:.819}, {0:.115, 1:.885},
			  								{0:.169, 1:.831},
			  								{0:.12, 1:.88}, {0:.11, 1:.89}])


	'''

	'''
	#score = ["f1"]:accuracy .74673,  precision  .29250, recall  .63400  
	#algorithm run when my feature is in the feature list
	params = dict(select__k = [1, 2, 3, 4],
				select__score_func = [f_classif, chi2],
				#reduce__n_components = [1, 2],
			  	classifier__C = [.95, .97],
			  	classifier__gamma = [1.41, 1.425, 1.43, 1.435],
			  	classifier__kernel = ["rbf"],
			  	classifier__class_weight = [{0:.20, 1:.80}, {0:.15, 1:.85},
			  								{0:.189, 1:.809}, {0:.14, 1:.86},
			  								{0:.179, 1:.819}, {0:.115, 1:.885},
			  								{0:.169, 1:.831},
			  								{0:.12, 1:.88}, {0:.11, 1:.89}])

	'''
	

	my_clf = GridSearchCV(pipe, params, cv = sss, scoring = K)
	my_clf.fit(features, labels)
	clf = my_clf.best_estimator_
	print params
	print my_clf.best_params_


	print test_classifier(clf, my_dataset, features_list)


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
	#print "Selected Features:", my_clf.best_params_["select__k"]


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)