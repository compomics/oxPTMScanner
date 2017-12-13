"""
Copyright 2017 Robbin Bouwmeester

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
This software can be used to flag MS2 spectra based on belonging to one 
of two classes of spectra. The two classes this code has been tested on 
oxPTM VS native spectra. The code can be used to train a new model that 
discriminates between two classes. After a model has been trained
the code can be used to make predictions for unseen spectra.

Library versions:

Python 2.7.13
xgboost.__version__ = '0.6'
sklearn.__version__ = '0.19.0'
scipy.__version__ = '0.19.1'
numpy.__version__ = '1.13.3'
pandas.__version__ = '0.20.3'

This project was made possible by MASSTRPLAN. MASSTRPLAN received funding 
from the Marie Sklodowska-Curie EU Framework for Research and Innovation 
Horizon 2020, under Grant Agreement No. 675132.
"""

# TODO make webversion

__author__ = "Robbin Bouwmeester"
__copyright__ = "Copyright 2017"
__credits__ = ["Robbin Bouwmeester","Demet Turan","Prof. Lennart Martens"]
__license__ = "Apache License, Version 2.0"
__version__ = "1.0"
__maintainer__ = "Robbin Bouwmeester"
__email__ = "Robbin.bouwmeester@ugent.be"
__status__ = "Beta version; ready for in-field application"

#Native library
import argparse
import pickle
import copy
from operator import itemgetter
from itertools import combinations

#Pandas
import pandas as pd

#Numpy
import numpy as np
from numpy.random import ranf
import numpy as np
import numpy.random as np_random

#Matplotlib
import matplotlib.pyplot as plt

#SciPy
import scipy.stats as st
import scipy

#ML
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import auc
import xgboost as xgb

def parse_msp(msp_entry,tic_normalization=True,min_perc=False,windowed_mode=False,top=10,window_size=100):
	"""
	Parse an MSP entry and return the identifier, peaks and intensity. Normalization 
	can be enabled and filtering on the intensity of peaks.

    Parameters
    ----------
    msp_entry : list
        list with the lines of the MSP entry
    tic_normalization : bool
        Return tic normalized intensities
	min_perc : bool
        Flag to use a minimal percentage intensity to filter peaks
	windowed_mode : bool
        Flag to use windowed mode to return the top intensity peaks
	top : int
        The top intensity peaks to filter (in windowed mode it will return the top peaks within the window)
	window_size : int
        The size of the window in windowed mode
		
    Returns
    -------
    str
		identifier of the msp entry
	list
		the m/z values in the msp entry
    list
		the intensity values in the msp entry    
    """
	
	identifier = ""
	mz_list = []
	intensity_list = []
	if tic_normalization: tot_tic = 0.0
	
	#Iterate over the lines in the MSP entry and record the identifiers, m/z and intensities
	for line in msp_entry:
		line = line.rstrip()
		if line == "": continue
		if line.startswith("Name: "):
			identifier = line.lstrip("Name: ").replace(",","_")
			continue
		if ":" in line: continue
		
		splitline = line.split("\t")

		mz_list.append(float(splitline[0]))
		intensity_list.append(float(splitline[1]))
		if tic_normalization: tot_tic += intensity_list[-1]
	
	#In the case of tic normalization iterate over the values and divide by total intensity
	if tic_normalization:
		for index,intens in enumerate(intensity_list):
			intensity_list[index] = intens/tot_tic
	
	#Filter based on the top intensities
	gr_mz_list,gr_intensity_list = get_top_spec(mz_list,
												intensity_list,
												min_perc=min_perc,
												windowed_mode=windowed_mode,
												top=top,
												window_size=window_size)
	
	return(identifier,gr_mz_list,gr_intensity_list)
	
def parse_mgf(mgf_entry,tic_normalization=True,min_perc=False,top=10,window_size=100,windowed_mode=False):
	"""
	Parse an MGF entry and return the identifier, peaks and intensity. Normalization 
	can be enabled and filtering on the intensity of peaks.

    Parameters
    ----------
    mgf_entry : list
        list with the lines of the MGF entry
    tic_normalization : bool
        Return tic normalized intensities
	min_perc : bool
        Flag to use a minimal percentage intensity to filter peaks
	windowed_mode : bool
        Flag to use windowed mode to return the top intensity peaks
	top : int
        The top intensity peaks to filter (in windowed mode it will return the top peaks within the window)
	window_size : int
        The size of the window in windowed mode
		
    Returns
    -------
    str
		identifier of the MGF entry
	list
		the m/z values in the MGF entry
    list
		the intensity values in the MGF entry    
    """
	
	identifier = ""
	mz_list = [0.0]
	intensity_list = [1.0]
	if tic_normalization: tot_tic = 0.0
	
	#Iterate over the lines in the MGF entry
	for line in mgf_entry:
		line = line.rstrip()
		if line == "": continue
		if line.startswith("TITLE="):
			identifier = line.lstrip("TITLE=").replace(",","_")
			continue
		if "=" in line: continue
		
		if "\t" in line: splitline = line.split("\t")
		else: splitline = line.split(" ")
		
		mz_list.append(float(splitline[0]))
		intensity_list.append(float(splitline[1]))
		if tic_normalization: tot_tic += intensity_list[-1]
	
	#In the case of tic normalization iterate over the values and divide by total intensity
	if tic_normalization:
		for index,intens in enumerate(intensity_list):
			intensity_list[index] = intens/tot_tic
	
	#Filter based on the top intensities
	gr_mz_list,gr_intensity_list = get_top_spec(mz_list,
												intensity_list,
												min_perc=min_perc,
												windowed_mode=windowed_mode,
												top=top,
												window_size=window_size)
	
	return(identifier,gr_mz_list,gr_intensity_list)

def get_top_spec(mz_list,intensity_list,min_perc=False,windowed_mode=False,top=10,window_size=100,add_dummy_peak=True):
	"""
	Filter in multiple ways on the intensity of peaks.

    Parameters
    ----------
    mz_list : list
        The m/z values of a spectrum in a list; equal length to the intensity list
    intensity_list : list
        The intensity values of a spectrum in a list; equal length to the m/z list
	min_perc : bool
        Flag to use a minimal percentage intensity to filter peaks
	windowed_mode : bool
        Flag to use windowed mode to return the top intensity peaks
	top : int
        The top intensity peaks to filter (in windowed mode it will return the top peaks within the window)
	window_size : int
        The size of the window in windowed mode
	add_dummy_peak : bool
		Flag to add a dummy peak at 0.0 m/z
	
    Returns
    -------
	list
		the filtered m/z values from the spectrum
    list
		the filtered intensity values from the spectrum  
    """
	gr_intensity_list = []
	gr_mz_list = []
	
	#In the case of minimal percentage... calculate perc intensity and filter
	if min_perc:
		for i,mz in zip(intensity_list,mz_list):
			if i > min_perc:
				gr_intensity_list.append(i)
				gr_mz_list.append(mz)
	
	#In the case of windowed mode... iterate over the possible windows and intensity values; take the top per window
	if windowed_mode:
		start_index = 0
		for w in range(window_size,int(max(mz_list)),window_size):
			temp_mz = []
			temp_intens = []
			temp_start_index = 0
			
			#Iterate over all m/z values and see if they fall within the window
			for mz,intens in zip(mz_list[start_index:],intensity_list[start_index:]):
				if mz > w and mz <= w+window_size:
					temp_start_index += 1
					temp_mz.append(mz)
					temp_intens.append(intens)
				if mz > w+window_size:
					break
			#Next window ignore all these lower values
			start_index = copy.deepcopy(temp_start_index)
			
			#Use all if there are less peaks than the top number of peaks it should select
			if len(temp_mz) <= top:
				gr_mz_list.extend(temp_mz)
				gr_intensity_list.extend(temp_intens)
				continue
			
			#Get the indexes of the top peaks
			idxs = np.sort(np.argpartition(np.array(temp_intens), -top)[-top:])
			gr_mz_list.extend([temp_mz[idx] for idx in idxs])
			gr_intensity_list.extend([temp_intens[idx] for idx in idxs])
	
	#If not windowed or min perc use a simple top peaks
	if not windowed_mode and not min_perc:
		if len(intensity_list) > top:
			#Get the indexes of the top peaks
			idxs = np.sort(np.argpartition(np.array(intensity_list), -top)[-top:])
			gr_mz_list = [mz_list[idx] for idx in idxs]
			gr_intensity_list = [intensity_list[idx] for idx in idxs]
		else:
			#If there are less peaks than top peaks; return all
			gr_mz_list = mz_list
			gr_intensity_list = intensity_list
	
	#If needed add a dummy peak; this is important later since I want to take into account immonium ions and small fragments
	if add_dummy_peak:
		gr_mz_list.insert(0,0.0)
		gr_intensity_list.insert(0,1.0)
	
	return(gr_mz_list,gr_intensity_list)
	
def get_feats(mz_list,intensity_list,feat_matrix,instance_index,feats,max_dist=275,allowed_c=[]):
	"""
	Retrieve features and write them to a matrix.

    Parameters
    ----------
    mz_list : list
        The m/z values of a spectrum in a list; equal length to the intensity list
    intensity_list : list
        The intensity values of a spectrum in a list; equal length to the m/z list
	feat_matrix : lil or csr matrix
        Sparse matrix that should be used to fill in the features from the m/z and intensity lists
	instance_index : int
        Row number in the matrix where the features should be filled in (indexing starts at 0)
	feats : list
        The bins used for features; should be sorted!; features are assigned if value is in between values of this list
	max_dist : int
        Maximum distance between peaks
	allowed_c : list
		Allowed bins (selected features); not used any more
	
    Returns
    -------
	matrix
		A sparse matrix (lil or csr) is returned with filled in features
    """
	# UNCOMMENT var below if standard library combinations is used
	#allowed_c = set(allowed_c)
	
	spectrum = zip(mz_list,intensity_list)
	dists_mz = []
	dists_mz_intens = []
	prev_analyzed = set()
	
	#Make deepcopy since we are going to change the spectra!
	spec_one = copy.deepcopy(spectrum)
	spec_two = copy.deepcopy(spectrum)
	
	#Iterate over the peaks and measure the distance in m/z between all combinations
	for peak_one in spec_one:
		if len(spec_two) == 1: continue
		spec_two = spec_two[1:]
		for peak_two in spec_two:
			dist_mz = abs(peak_one[0]-peak_two[0])
			if dist_mz > max_dist: break
			dists_mz.append(dist_mz)
			dists_mz_intens.append(peak_one[1]+peak_two[1])
	
	# UNCOMMENT code below if standard library combinations is used
	#for c in combinations(spectrum,2):
	#	dist_mz = abs(c[0][0]-c[1][0])
	#	if c[0][0] in prev_analyzed: continue
	#	if dist_mz > max_dist: 
	#		prev_analyzed.add(c[0][0])
	#		continue
	#	if len(allowed_c) != 0:
	#		if dist_mz not in allowed_c: continue
	#	dists_mz.append(dist_mz)
	#	dists_mz_intens.append(c[0][1]+c[1][1])
	
	#Digitize the delta m/z; assign bins for all delta m/z s
	index_bins = np.digitize(dists_mz,feats)
	
	#Iterate over assigned bins and sum the intensity for possible existing values
	print(feat_matrix.shape)
	for index,intens in zip(index_bins,dists_mz_intens):
		feat_matrix[instance_index,index-1] += intens

	return(feat_matrix)
	
def read_msp(infile_name,feat_lim_file="",
			 sum_feats=False,selected_features=[],
			 max_dist=275,step_size=0.005,feat_bins=[],
			 top_peaks=50,windowed_mode=False):
	"""
	Read an MSP file and put the features into a matrix.

    Parameters
    ----------
    infile_name : list
        The infile MSP file.
    feat_lim_file : list
        Old variable with the name of a file that contains the features.
	sum_feats : bool
        Old variable used to sum features of the two classes.
	selected_features : list
        Old variable for selected features; use feat_bins.
	max_dist : int
        Maximum distance between peaks
	step_size : float
		Size between the m/z values for bins.
	feat_bins : list
		Bins to use for features.
	top_peaks : int
		Number of peaks to select based on the intensity
	windowed_mode : bool
		Flag to used windowed mode for selecting the highest intensity peaks
		
    Returns
    -------
	matrix
		A sparse matrix (csr) is returned with filled in features
	list
		Used features for binning
	list
		Identifiers of all entries in the MSP file
	int
		Number of analyzed MSP entries
    """

	#infile = open(infile_name)
	infile = infile_name.readlines()

	if len(feat_lim_file) > 0:
		selected_features = [float(f.strip()) for f in open(feat_lim_file).readlines()]
		
	counter = 0
	temp_entry = []
	instance_names = []
	num_instances = num_instances_msp(infile)
	#print(num_instances)

	if len(feat_bins) == 0: feat_bins = np.arange(0,max_dist+step_size,step_size)
	
	#Initialize the feature matrix, must be lil since scr is slow when mutating values!
	feat_matrix = scipy.sparse.lil_matrix((num_instances, len(feat_bins)),dtype=np.float32)
	
	#Iterate over the file and filter out single entries
	for line in infile:
		if line.startswith("Name: "):
			if len(temp_entry) == 0:
				temp_entry.append(line.strip())
				continue
			#For this entry get identifier,m/z,intensities
			identifier,mz_list,intensity_list = parse_msp(temp_entry,top=top_peaks,windowed_mode=windowed_mode)
			instance_names.append(identifier)
			#Fill in the feature matrix
			feat_matrix = get_feats(mz_list,intensity_list,feat_matrix,counter,feat_bins,allowed_c=selected_features,max_dist=max_dist)
			
			#Make sure the current line is still used for the next entry
			temp_entry = [line]
			
			#print(counter)
			counter += 1
			
		temp_entry.append(line.strip())
	
	#If everything is empty; return
	if len(temp_entry) == 0:
		temp_entry.append(line.strip())
		return(feat_matrix.asformat("csr"),feat_bins,instance_names,counter)

	#Analyse the last record; since we do not know when the spectra ends
	identifier,mz_list,intensity_list = parse_msp(temp_entry,top=top_peaks,windowed_mode=windowed_mode)
	instance_names.append(identifier)
	feat_matrix = get_feats(mz_list,intensity_list,feat_matrix,counter,feat_bins,allowed_c=selected_features)
	
	#print(counter)
	counter += 1
	
	return(feat_matrix.asformat("csr"),feat_bins,instance_names,counter)
	
def num_instances_msp(infile_name):
	"""
	Count the number of entries in the MSP file.

    Parameters
    ----------
    infile_name : list
        The infile MSP file.

    Returns
    -------
	int
		Number of analyzed MSP entries
    """
	infile = infile_name
	num_instances = 0
	for line in infile:
		if line.startswith("Name: "):
			num_instances += 1
	return(num_instances)

def read_mgf(infile_name,feat_lim_file="",
			 sum_feats=False,selected_features=[],
			 max_dist=275,step_size=0.005,feat_bins=[],
			 top_peaks=50,windowed_mode=False):
	"""
	Read an MGF file and put the features into a matrix.

    Parameters
    ----------
    infile_name : list
        The infile MGF file.
    feat_lim_file : list
        Old variable with the name of a file that contains the features.
	sum_feats : bool
        Old variable used to sum features of the two classes.
	selected_features : list
        Old variable for selected features; use feat_bins.
	max_dist : int
        Maximum distance between peaks
	step_size : float
		Size between the m/z values for bins.
	feat_bins : list
		Bins to use for features.
	top_peaks : int
		Number of peaks to select based on the intensity
	windowed_mode : bool
		Flag to used windowed mode for selecting the highest intensity peaks
		
    Returns
    -------
	matrix
		A sparse matrix (csr) is returned with filled in features
	list
		Used features for binning
	list
		Identifiers of all entries in the MGF file
	int
		Number of analyzed MGF entries
    """		 
	
	#infile = open(infile_name)
	infile = infile_name.readlines()
	
	if len(feat_lim_file) > 0:
		selected_features = [float(f.strip()) for f in open("selected_features.txt").readlines()]
		
	counter = 0
	temp_entry = []
	instance_names = []
	num_instances = num_instances_mgf(infile)
	#print(num_instances)

	if len(feat_bins) == 0: feat_bins = np.arange(0,max_dist+step_size,step_size)
	
	#Initialize the feature matrix, must be lil since scr is slow when mutating values!
	feat_matrix = scipy.sparse.lil_matrix((num_instances, len(feat_bins)),dtype=np.float32)
	
	#Iterate over the file and filter out single entries
	for line in infile:
		if line.startswith("END IONS"):
			#For this entry get identifier,m/z,intensities
			identifier,mz_list,intensity_list = parse_mgf(temp_entry,top=top_peaks,windowed_mode=windowed_mode)
			instance_names.append(identifier)
			#Fill in the feature matrix
			feat_matrix = get_feats(mz_list,intensity_list,feat_matrix,counter,feat_bins,allowed_c=selected_features,max_dist=max_dist)
			counter += 1
			#print(counter)
			temp_entry = []
			continue
		if line.startswith("BEGIN IONS"):
			continue
		temp_entry.append(line)

	return(feat_matrix.asformat("csr"),feat_bins,instance_names,counter)

def num_instances_mgf(infile_name):
	"""
	Count the number of entries in the MGF file.

    Parameters
    ----------
    infile_name : list
        The infile MGF file.

    Returns
    -------
	int
		Number of analyzed MGF entries
    """
	infile = infile_name
	num_instances = 0
	for line in infile:
		if line.startswith("BEGIN IONS"):
			num_instances += 1
	return(num_instances)	

def train_xgb(X,y):
	"""
	Train an XGBoost model with hyper parameter optimization.

    Parameters
    ----------
    X : matrix
        Matrix with all the features, every instance should be coupled to the y-value
    y : vector
        Vector with the class, every value should be coupled to an x-vector with features
		
    Returns
    -------
	object
		Trained XGBoost model
	object
		Cross-validation results
    """
	
	xgb_handle = xgb.XGBClassifier()

	one_to_left = st.beta(10, 1)  
	from_zero_positive = st.expon(0, 50)
	
	#Define distributions to sample from for hyper parameter optimization
	param_dist = {  
	    "n_estimators": st.randint(3, 40),
	    "max_depth": st.randint(3, 40),
	    "learning_rate": st.uniform(0.05, 0.4),
	    "colsample_bytree": one_to_left,
	    "subsample": one_to_left,
	    "gamma": st.uniform(0, 10),
	    "reg_alpha": from_zero_positive,
	    "min_child_weight": from_zero_positive,
	}

	n_iter_search = 20
	random_search = RandomizedSearchCV(xgb_handle, param_distributions=param_dist,
	                                   n_iter=n_iter_search,verbose=10,scoring="roc_auc",
	                                   n_jobs=1,cv=5)

	random_search_res_xgb = random_search.fit(X, y)
	
	#Get the best model that was retrained on all data
	xgb_model = random_search_res_xgb.best_estimator_

	return(xgb_model,random_search_res_xgb)
	
def train_xgb_lim(X,y,params_dist,out_dir="res/"):
	"""
	Train an XGBoost model with set hyper parameters.

    Parameters
    ----------
    X : matrix
        Matrix with all the features, every instance should be coupled to the y-value
    y : vector
        Vector with the class, every value should be coupled to an x-vector with features
	params_dist : dict
		The hyperparameters to use
	out_dir : str
		String value that points to a directory used for output
		
    Returns
    -------
	list
		The cross-validated predictions.
    """
	#There is a need to unpack the hyperparameter dictionary with "**"
	xgb_handle = xgb.XGBClassifier(**params_dist)
	#Using predict_proba since ROC-curve
	test_preds = cross_val_predict(xgb_handle,X,y,method="predict_proba")
	plot_roc(X,y,test_preds[:,1],fname=out_dir+"roc.png")
	return(test_preds)

def plot_feat_imp(feats_index,feat_names,X,y,top_imp=10,out_dir="res/"):
	"""
	Plot the most important features in a boxplot and seperate on class (y)

    Parameters
    ----------
	feats_index : list
		Indexes of the features coupled to the X matrix
	feat_names : list
		Names of the features coupled to specific indices
    X : matrix
        Matrix with all the features, every instance should be coupled to the y-value
    y : vector
        Vector with the class, every value should be coupled to an x-vector with features
	top_imp : int
		Plot this number of top features in a boxplot
	out_dir : str
		String value that points to a directory used for output
		
    Returns
    -------
	
    """
	for fi in feats_index[0:top_imp]:
		#Need for a dense matrix when plotting
		plt.boxplot([X.todense()[y==1,:][:,fi],X.todense()[y==0,:][:,fi]])
		plt.title(feat_names[fi])
		plt.savefig(out_dir+"%s_feat_groups.png" % (feat_names[fi]), bbox_inches='tight')
		plt.close()

def plot_train_distr(xgb_model,X,y,out_dir="res/"):
	"""
	Plot probability distributions for the input matrix.

    Parameters
    ----------
	xgb_model : object
		Trained XGBoost model
    X : matrix
        Matrix with all the features, every instance should be coupled to the y-value
    y : vector
        Vector with the class, every value should be coupled to an x-vector with features
	out_dir : str
		String value that points to a directory used for output
		
    Returns
    -------

    """
	
	#Get the predicted probabilities for both classes (store them seperately)
	probs_oxid = xgb_model.predict_proba(X[y==1])[:,1]
	probs_native = xgb_model.predict_proba(X[y==0])[:,1]
	
	#Plot density distribution for probailities
	pd.Series(probs_oxid).plot(kind="density")
	pd.Series(probs_native).plot(kind="density")
	axes = plt.gca()
	axes.set_xlim([0.0,1.0])
	plt.savefig(out_dir+"density_groups.png", bbox_inches='tight')
	plt.close()
	
	#Plot density distribution for probailities; zoom in more so the y-axis is readable
	pd.Series(probs_oxid).plot(kind="density")
	pd.Series(probs_native).plot(kind="density")
	axes = plt.gca()
	axes.set_xlim([0.0,1.0])
	axes.set_ylim([0.0,1.0])
	plt.savefig(out_dir+'density_groups_zoomed.png', bbox_inches='tight')
	plt.close()

	#Plot probability distributions in histogram
	plt.hist(probs_native,bins=100)
	plt.hist(probs_oxid,bins=100)
	plt.savefig(out_dir+'hist_groups.png', bbox_inches='tight')
	plt.close()
	
	#Plot probability distributions in histogram; zoom in more so the y-axis is readable
	plt.hist(probs_native,bins=100)
	plt.hist(probs_oxid,bins=100)
	axes = plt.gca()
	axes.set_ylim([0.0,1000.0])
	plt.savefig(out_dir+'hist_groups_zoomed.png', bbox_inches='tight')
	plt.close()

def xgboost_to_wb(xgboost,outfile="model.pickle"):
	"""
	Pickle a trained XGBoost model.

    Parameters
    ----------
	xgboost : object
		Trained XGBoost model
    outfile : str
        Location of the pickle
		
    Returns
    -------

    """
	pickle.dump(xgboost, open(outfile,"wb"))

def plot_roc(X,y,test_preds,fname="res/roc.png"):
	"""
	Plot an ROC-curve and write to a file

    Parameters
    ----------
    X : matrix
        Matrix with all the features, every instance should be coupled to the y-value
    y : vector
        Vector with the class, every value should be coupled to an x-vector with features
	test_preds : vector
		Predicted probabilities for classes

    Returns
    -------

    """
	#Retrieve multiple fpr and tpr values for different thresholds
	fpr, tpr, thresholds = roc_curve(y,test_preds)
	plt.plot(fpr, tpr)
	plt.title(auc(fpr, tpr))
	plt.savefig(fname, bbox_inches='tight')
	plt.close()

def get_min_diff(zero_f="NIST/human_hcd_synthetic_oxidized.msp",
				 one_f="NIST/human_hcd_synthetic_native.msp",
				 outfile="res_small/selected_features_diff.txt",
				 top_mean = 1000,
				 top_peaks = 50,
				 max_distance = 275,
				 distance_bins = 0.005,
				 windowed_mode=False):
	"""
	Function that is used to get the most important bins in terms of mean intensity between the two classes.

    Parameters
    ----------
	zero_f : str
		Filename (and dir) that contains the spectra for class 0
	one_f : str
		Filename (and dir) that contains the spectra for class 1
    top_mean : int
        Return this amount of features (bins)
    top_peaks : int
        The top intensity peaks to filter (in windowed mode it will return the top peaks within the window)
	max_distance : int
        Maximum distance in m/z to use for features
	windowed_mode : bool
        Flag to use windowed mode to return the top intensity peaks
	distance_bins : float
        Distance in m/z between the bins

    Returns
    -------
	list
		The most important bins that are on average different between the classes
    """
	
	#Check the file extension and parse to get features for class zero
	if zero_f.endswith(".mgf"): feats_zero_sum,feat_bins,instance_names,count_zero = read_mgf(zero_f,
																							  sum_feats=True,
																							  max_dist=max_distance,
																							  step_size=distance_bins,
																							  top_peaks=top_peaks)
	elif zero_f.endswith(".msp"): feats_zero_sum,feat_bins,instance_names,count_zero = read_msp(zero_f,
																								sum_feats=True,
																								max_dist=max_distance,
																								step_size=distance_bins,
																								top_peaks=top_peaks)
	else: return(False)
	
	#Check the file extension and parse to get features for class one
	if one_f.endswith(".mgf"): feats_one_sum,feat_bins,instance_names,count_one = read_mgf(one_f,
																						   sum_feats=True,
																						   max_dist=max_distance,
																						   step_size=distance_bins,
																						   top_peaks=top_peaks)
	elif one_f.endswith(".msp"): feats_one_sum,feat_bins,instance_names,count_one = read_msp(one_f,
																							 sum_feats=True,
																							 max_dist=max_distance,
																							 step_size=distance_bins,
																							 top_peaks=top_peaks)
	else: return(False)
	
	#Get average difference between the bins of both groups
	diffs = [abs(m1-m2) for m1,m2 in zip(feats_zero_sum.mean(axis=0).tolist()[0],feats_one_sum.mean(axis=0).tolist()[0])]
	
	#Get the indexes of the biggest differences in bins
	indexes_diff = sorted(list(enumerate(diffs)),key=itemgetter(1),reverse=True)
	selected_features_diff = [feat_bins[ind] for ind,val in indexes_diff[0:top_mean]]
	selected_features_diff.sort()
	
	#For the important bins we need the next number to create a closed bin; calculated vals; extend to bins
	diff_bins = [sfd+distance_bins for sfd in selected_features_diff]
	diff_bins.extend(selected_features_diff)
	diff_bins.sort()
	
	#Remove duplicate values
	diff_bins = list(set(diff_bins))
	diff_bins.sort()
	
	#Write feats to a file
	outfile_feats = open(outfile,"w")
	outfile_feats.write("\n".join(map(str,diff_bins)))
	outfile_feats.close()

	return(diff_bins)
	
def train_initial_classifier(zero_f="NIST/human_hcd_synthetic_oxidized.msp",
							 one_f="NIST/human_hcd_synthetic_native.msp",
							 selected_features_diff=[],
							 top_mean = 1000,
							 top_peaks = 100,
							 max_distance = 275,
							 distance_bins = 0.005,
							 windowed_mode = False,
							 out_dir="res/"):
	"""
	Function that is used to train an XGBoost model to discrimate between the MS2 spectra from two files.

    Parameters
    ----------
	zero_f : str
		Filename (and dir) that contains the spectra for class 0
	one_f : str
		Filename (and dir) that contains the spectra for class 1
	selected_features_diff : list
		List with m/z bin values that should be used as features
    top_mean : int
        Return this amount of features (bins)
    top_peaks : int
        The top intensity peaks to filter (in windowed mode it will return the top peaks within the window)
	max_distance : int
        Maximum distance in m/z to use for features
	windowed_mode : bool
        Flag to use windowed mode to return the top intensity peaks
	distance_bins : float
        Distance in m/z between the bins
	out_dir : str
		Directory to write the results to
		
    Returns
    -------
	dict
		Used parameters in the XGBoost model
	list
		Most important features according to the F-score in the XGBoost model
    """
	#Check the file extension and parse to get features for class zero
	if zero_f.endswith(".mgf"): feats_zero,feat_bins,instance_names,count_zero = read_mgf(zero_f,sum_feats=False,
																						  feat_bins=selected_features_diff,
																						  max_dist=max_distance,
																						  step_size=distance_bins,
																						  top_peaks=top_peaks)
	elif zero_f.endswith(".msp"): feats_zero,feat_bins,instance_names,count_zero = read_msp(zero_f,
																							sum_feats=False,
																							feat_bins=selected_features_diff,
																							max_dist=max_distance,
																							step_size=distance_bins,
																							top_peaks=top_peaks)
	else: return(False) # TODO display error!
	
	#Check the file extension and parse to get features for class one
	if one_f.endswith(".mgf"): feats_one,feat_bins,instance_names,count_one = read_mgf(one_f,
																					   sum_feats=False,
																					   feat_bins=selected_features_diff,
																					   max_dist=max_distance,
																					   step_size=distance_bins,
																					   top_peaks=top_peaks)
	elif one_f.endswith(".msp"): feats_one,feat_bins,instance_names,count_one = read_msp(one_f,
																						 sum_feats=False,
																						 feat_bins=selected_features_diff,
																						 max_dist=max_distance,
																						 step_size=distance_bins,
																						 top_peaks=top_peaks)
	else: return(False) # TODO display error!

	#Prepare labels equal to length class zero and one
	y = [0]*(count_zero)
	y.extend([1]*(count_one))

	y = np.array(y)
	
	#Stack the feature matrices of both classes
	X = scipy.sparse.vstack((feats_zero,feats_one))
	
	#Train optimizing the hyperparameters
	xgb_model,random_search_res_xgb = train_xgb(X,y)
	#print(random_search_res_xgb.best_params_)
	#print(random_search_res_xgb.best_score_)
	
	#Train use selected hyperparameters
	train_xgb_lim(X,y,random_search_res_xgb.best_params_,out_dir=out_dir)
	plot_train_distr(xgb_model,X,y,out_dir=out_dir)
	
	#Flush to pickle
	xgboost_to_wb(random_search_res_xgb,outfile=out_dir+"model.pickle")
	
	random_search_res_xgb = pickle.load(open(out_dir+"model.pickle","rb"))
	
	#Plot some of the feature importances and probs
	fscores = xgb_model.booster().get_fscore()
	fscores_list = sorted(list(fscores.items()),key=itemgetter(1),reverse=True)
	selected_features_indexes = map(int,[f.replace("f","") for f,n in fscores_list])
	selected_features_xgboost  = [selected_features_diff[sfp] for sfp in selected_features_indexes]
	plot_feat_imp(selected_features_indexes,selected_features_diff,X,y,out_dir=out_dir)
	
	return(random_search_res_xgb.best_params_,selected_features_xgboost)
	
def apply_model(infile_pred,
				infile_model,
				infile_features,
				filename,
				threshold_prob=0.5,
				windowed_peak_picking=False,
				out_dir="res/",
				top_peaks=50,
				max_distance=275,
				distance_bins=0.005):
	"""
	Make predictions to discriminate between classes of spectra using pretrained models.

    Parameters
    ----------
	infile_pred : str
		Filename (and dir) that contain the spectra we need to make predictions for
	infile_model : str
		Filename (and dir) that points to the trained XGBoost model
	infile_features : str
		Filename (and dir) that contains the m/z bins (features)
    top_peaks : int
        The top intensity peaks to filter (in windowed mode it will return the top peaks within the window)
	max_distance : int
        Maximum distance in m/z to use for features
	windowed_peak_picking : bool
        Flag to use windowed mode to return the top intensity peaks
	distance_bins : float
        Distance in m/z between the bins
	threshold_prob : float
		Threshold to determine an instance belongs to class zero or one
	out_dir : str
		Directory to write the results to
		
    Returns
    -------

    """
	
	#Read the to be used m/z bins (features)
	features = [f.strip() for f in open(infile_features).readlines()]
	
	#Check the file extension and parse to get features
	if filename.endswith(".mgf"): 
		feats,feat_bins,instance_names,count_inst = read_mgf(infile_pred,
															 sum_feats=False,
															 feat_bins=features,
															 max_dist=max_distance,
															 step_size=distance_bins,
															 top_peaks=top_peaks)
	elif filename.endswith(".msp"): 
		feats,feat_bins,instance_names,count_inst = read_msp(infile_pred,
															 sum_feats=False,
															 feat_bins=features,
															 max_dist=max_distance,
															 step_size=distance_bins,
															 top_peaks=top_peaks)
	else: return(False)
	
	#print(feats.shape)
	#print(feats)
	#print(len(instance_names))
	#print(instance_names)
	
	#Load the XGBoost model
	random_search_res_xgb = pickle.load(open(infile_model,"rb"))
	
	#Format the preds
	preds = pd.DataFrame(random_search_res_xgb.predict_proba(feats),index=instance_names,columns=["Prob_class_0","Prob_class_1"])
	pd.Series(preds["Prob_class_1"]).plot(kind="density")
	axes = plt.gca()
	axes.set_xlim([0.0,1.0])
	axes.set_xlabel("Probability of oxidation in spectrum")
	return(plt,preds["Prob_class_1"])

def parse_argument():
	"""
	Read arguments from the command line

    Parameters
    ----------
		
    Returns
    -------

    """
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--top_peaks", type=int, dest="top_peaks", default=50,
						help="Number of peaks to extract and consider for combinations in a spectrum")
	
	parser.add_argument("--top_mean", type=int, dest="top_mean", default=2000,
						help="The top bins in different mean between group A and B to learn on")
						
	parser.add_argument("--max_distance", type=int, dest="max_distance", default=275,
						help="The maximum difference between peaks (maximum bin value)")
	
	parser.add_argument("--distance_bins", type=float, dest="distance_bins", default=0.005,
						help="Distance in m/z of the bins")
	
	parser.add_argument("--file_a", type=str, dest="file_a",default="NIST/human_hcd_synthetic_native.msp",
						help="The mgf or msp of group A")
	
	parser.add_argument("--file_b", type=str, dest="file_b", default="NIST/human_hcd_synthetic_oxidized.msp",
						help="The mgf or msp of group B")
	
	parser.add_argument("--file_pred", type=str, dest="file_pred", default="NIST/human_hcd_synthetic_native.msp",
						help="The mgf or msp to make predictions for")
						
	parser.add_argument("--out_dir", type=str, dest="out_dir", default="res/",
						help="Directory where the results are written. WILL OVERWITE EXISTING FILES!")

	parser.add_argument("--make_pred", action="store_true",
						help="Flag that can be included to indicate predictions are desired instead of training a model")
						
	parser.add_argument("--windowed_peak_picking", action="store_true",
						help="Flag that can be included to use windowed peak picking per 100 m/z")

	parser.add_argument("--model", type=str, dest="model", default="res/model.pickle",
						help="File that refers to a model that is used for predictions")
	
	parser.add_argument("--feats", type=str, dest="feats", default="res/selected_features_diff.txt",
						help="File that refers to the features that are used in the model")

	parser.add_argument("--version", action="version", version="%(prog)s 1.0")

	results = parser.parse_args()

	return(results)

def main():
	#Get command line arguments
	argu = parse_argument()
	
	#Train a new model
	if not argu.make_pred:
		selected_features_diff = get_min_diff(zero_f=argu.file_a,
											  one_f=argu.file_b,
											  outfile=argu.out_dir+"/selected_features.txt",
											  top_peaks=argu.top_peaks,
											  top_mean=argu.top_mean,
											  max_distance=argu.max_distance,
											  distance_bins=argu.distance_bins,
											  windowed_mode=argu.windowed_peak_picking)
		# UNCOMMENT line below and comment above call to bypass initial feature selection
		#selected_features_diff = [float(f.strip()) for f in open("res/selected_features.txt").readlines()]

		random_search_params,selected_features_xgb = train_initial_classifier(zero_f=argu.file_a,
																			  one_f=argu.file_b,
																			  selected_features_diff=selected_features_diff,
																			  top_peaks=argu.top_peaks,
																			  top_mean=argu.top_mean,
																			  max_distance=argu.max_distance,
																			  distance_bins=argu.distance_bins,
																			  windowed_mode=argu.windowed_peak_picking)
	#Make predictions using existing trained model
	if argu.make_pred:
		apply_model(argu.file_pred,argu.model,argu.feats,out_dir=argu.out_dir,
					windowed_peak_picking=argu.windowed_peak_picking,
					top_peaks=argu.top_peaks,
					top_mean=argu.top_mean,
					max_distance=argu.max_distance,
					distance_bins=argu.distance_bins)

if __name__ == "__main__":
	main()