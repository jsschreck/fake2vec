import numpy as np
import pickle, sys, os, itertools

import matplotlib, pylab
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

def class_report(X, y, model, outfile = "class_report"):
	output_labels = {0: 'publisher', 1: 'fact', 2: 'bias'}
	y_pred = model.predict(X)
	for k,(_y, _y_pred) in enumerate(zip(y, y_pred)):
		report = classification_report(_y, _y_pred.round(), output_dict=True)
		report = pd.DataFrame(report).transpose()
		report.to_csv(outfile + ".{}.csv".format(output_labels[k]))

def plot_history(history, c1 = 'loss', c2 = 'pub_acc'):
	## As loss always exists
	epochs = range(1,len(history.history['loss']) + 1)
	colors = plt.cm.rainbow(np.linspace(0,1,4))

	## Loss plots ##########

	fig = fig_window(0,1,1)
	ax1 = fig.add_subplot(1,2,1)
	ax2 = fig.add_subplot(1,2,2)

	losses = ['loss', 'pub_loss', 'fact_loss', 'bias_loss']
	for l,c in zip(losses,colors):
		ax1.semilogy(epochs, history.history[l], color = c)
	ax1.legend([" ".join(x.split("_")) for x in losses], loc = 'best')
	ax1.set_title("Training split (70\%)")

	val_losses = ['val_' + x for x in losses]
	for l,c in zip(val_losses,colors):
		ax2.semilogy(epochs, history.history[l], color = c)
	ax2.legend([" ".join(x.split("_")) for x in val_losses], loc = 'best')
	ax2.set_title("Validation split (30\%)")

	for ax in [ax1,ax2]:
		ax.set_xlabel(r"Epochs")
		ax.set_ylabel(r"Loss")

	plt.tight_layout()
	PDF_SAVECHOP("results/training_loss")
	fig.clear()

	## Accuracy plots ##########
	fig = fig_window(1,1,1)
	ax1 = fig.add_subplot(1,2,1)
	ax2 = fig.add_subplot(1,2,2)

	accuracies = ['pub_acc', 'fact_acc', 'bias_acc']
	for l,c in zip(accuracies,colors):
		ax1.plot(epochs, history.history[l], color = c)
	ax1.legend([" ".join(x.split("_")) for x in accuracies], loc = 'best')
	ax1.set_title("Training split (70\%)")

	val_accuracies = ['val_' + x for x in accuracies]
	for l,c in zip(val_accuracies,colors):
		ax2.plot(epochs, history.history[l], color = c)
	ax2.legend([" ".join(x.split("_")) for x in val_accuracies], loc = 'best')
	ax2.set_title("Validation split (30\%)")

	for ax in [ax1,ax2]:
		ax.set_xlabel(r"Epochs")
		ax.set_ylabel(r"Accuracy")
		ax.set_ylim([0.4, 1.0])

	plt.tight_layout()
	PDF_SAVECHOP("results/training_accuracy")

def swap_cols(arr, frm, to):
	arr[:,[frm, to]] = arr[:,[to, frm]]

def swap_rows(arr, frm, to):
	arr[[frm, to],:] = arr[[to, frm],:]

def plot_confusion_matrix(model, X, y_true, encoders, normalize=False, cmap=plt.cm.Blues, title = 'train'):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	# publisher, fact, bias

	output_labels = {0: 'publisher', 1: 'fact', 2: 'bias'}

	class_list = [range(937), [0,2,1], [1,3,4,0,6,5,2]]

	print("Evaluating model on training data ...")
	y_pred = model.predict(X)
	print("... finished")

	for k, (_y_true, _y_pred, classes) in enumerate(zip(y_true, y_pred, class_list)):

		if not k: continue

		labels = classes
		classes = encoders[k].inverse_transform(classes)

		fig = fig_window(k,1,1)
		ax  = fig.add_subplot(1,1,1)

		print("Working on class {}".format(output_labels[k]))

		cm = confusion_matrix(_y_true.argmax(axis=1),
								_y_pred.round().argmax(axis=1),
								labels = labels)
		print("... finished building confusion matrix for class {}".format(output_labels[k]))

		print(cm)

		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

		img = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin = 0, vmax = 1)
		fig.colorbar(img)
		#ax.set_title(title)

		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.

		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			ax.text(j, i, format(cm[i, j], fmt),
			         horizontalalignment="center",
			         color="white" if cm[i, j] > thresh else "black")

		tick_marks = np.arange(len(classes))
		ax.set_xticks(tick_marks)
		ax.set_yticks(tick_marks)

		if k == 1:
			classes = ['HIGH', 'MIXED', 'LOW']
		if k == 2:
			classes = ['extreme-left', 'left', 'left-center', 'center', 'right-center', 'right', 'extreme-right']

		ax.set_xticklabels(classes, rotation = 45, ha="right")
		ax.set_yticklabels(classes)

		ax.set_ylabel('True label')
		ax.set_xlabel('Predicted label')

		if title == 'train':
			_title = 'Training split (70\%)'
		else:
			_title = 'Validation split (30\%)'

		ax.set_title(_title)

		plt.tight_layout()
		PDF_SAVECHOP("results/confusion_matrix_{}_{}".format(title, output_labels[k]))

if __name__ == '__main__':

	WEIGHTS_FPATH = "models/classifier/model.h5.pkl"

	with open(WEIGHTS_FPATH, "rb") as fid:
		classifier = pickle.load(fid)

	with open("data/plot_data.pkl", "rb") as fid:
		history, N_classes, X_train, y_input, X_test, y_input_test = pickle.load(fid)

	with open("data/label_encoders.pkl", "rb") as fid:
		encoders = pickle.load(fid)
	#print(history.history.keys())

	plot_history(history, c1 = 'loss', c2 = 'pub_acc')

	'''
	plot_confusion_matrix(classifier, X_test, y_input_test, encoders,
							normalize=True,
							cmap=plt.cm.Blues,
							title = 'test')
	'''
