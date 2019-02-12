import numpy as np
import pickle, sys, os, itertools  

import matplotlib, pylab
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from train_doc2vec import load_data

import pandas as pd
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def PDF_SAVECHOP(f,PNG=False):
	
	if PNG==False:
		pylab.savefig(f+'.pdf',
		        pad_inches=0,transparent=False)
		os.system('pdfcrop %s.pdf' % f)
		os.system('mv %s-crop.pdf %s.pdf' % (f,f))
	if PNG:
		pylab.savefig(f+'.jpg',
		        pad_inches=0)
		os.system('convert -trim %s.jpg %s.jpg' % (f,f))

def fig_window(_id = 0, scale_x = 1, scale_y = 1):
	plt.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'],
	                  'monospace':['Computer Modern Typewriter']})
	fig_width_pt = 252  # Get this from LaTeX using \showthe\columnwidth
	inches_per_pt = 1.0/72.27               # Convert pt to inch
	golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
	fig_width = 7 #fig_width_pt*inches_per_pt  # width in inches
	fig_height = fig_width*golden_mean      # height in inches
	fig_size =  [fig_width,fig_height]
	params = {'backend': 'ps',
	          'axes.labelsize': 12,
	          'font.size': 12,
	          'legend.fontsize': 10,
	          'legend.handlelength': 1,
	          'xtick.labelsize': 12,
	          'ytick.labelsize': 12,
	          #'axes.linewidth': 0.1,
	          'text.usetex': True,
	          'text.latex.preamble': [r"\usepackage{amstext}", r"\usepackage{mathpazo}"],
	          #'xtick.major.pad': 10,
	          #'ytick.major.pad': 10
	    }
	fig = plt.figure(_id, figsize = (scale_x * fig_width, scale_y * fig_height), facecolor='w', edgecolor='k')
	pylab.rcParams.update(params)
	return fig

def class_imbalance(df,label_index):

	label = ['publisher','fact','bias'][label_index]

	fig = fig_window(0,1,1)

	le = LabelEncoder()
	df[label] = le.fit_transform(df[label])
	
	if label_index == 0: 
		ax = sns.countplot(data=df, 
					   x=label, 
					   order = df[label].value_counts().index,
					   log=True)
	else:
		ax = sns.countplot(data=df, 
						   x=label, 
						   order = df[label].value_counts().index)
	ax.set_title("Articles per " + label + " class")
	
	if label_index == 0:
		ticks = range(0, max(df[label]))
		xticks = []
		for x in ticks:
			if (x%100) == 0:
				xticks.append(x)
			else:
				xticks.append(" ")
		ax.set_xticklabels(xticks)

	if label_index == 1:
		ax.set_xticklabels(le.inverse_transform(range(3)), rotation = 15, ha="right")
	if label_index == 2:
		ax.set_xticklabels(le.inverse_transform(range(7)), rotation = 15, ha="right")

	#fig.tight_layout()
	PDF_SAVECHOP("results/class_histogram_{}".format(label))

	#plt.show()

if __name__ == '__main__':
	
	with open("data/plot_data.pkl", "rb") as fid:
		classifier, history, N_classes, X_train, y_input, X_test, y_input_test = pickle.load(fid)
	
	with open("data/label_encoders.pkl", "rb") as fid:
		encoders = pickle.load(fid)
	
	if os.path.isfile("data/doc2vec_train.pkl"):
		with open("data/doc2vec_train.pkl", "rb") as fid:
			df = pickle.load(fid)
	else:
		doc2vec_train = 'data/doc2vec_train.csv'
		df, company_to_token, token_to_company = load_data(doc2vec_train = doc2vec_train)
		with open("data/doc2vec_train.pkl", "wb") as fid:
			pickle.dump(df,fid)
	df.rename(columns={'fact_score':'fact', 'bias_score':'bias'}, inplace=True)

	for label_index in [0,1,2]:
		class_imbalance(df,label_index)
	
	
