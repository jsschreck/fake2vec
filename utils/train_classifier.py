from sklearn.model_selection import train_test_split
from fake2vec.utils.text_processor import WordsContainer
from fake2vec.utils.train_doc2vec import load_data

import numpy as np
import pandas as pd
import sys, re, os, argparse, pickle, functools, collections, warnings

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from keras.utils import np_utils
from keras.regularizers import l1, l2
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn import utils
#, get_vectors

import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, BatchNormalization, MaxPooling1D
from keras.optimizers import SGD, Adam

warnings.filterwarnings("ignore", category=FutureWarning)

top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
top5_acc.__name__ = 'top5_acc'
top10_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=10)
top10_acc.__name__ = 'top10_acc'

def CountFrequency(arr):
    return collections.Counter(arr)

def get_callbacks(WEIGHTS_FPATH, LOG_FPATH, monitor):
	callbacks = [
		ModelCheckpoint(WEIGHTS_FPATH, monitor = monitor, save_best_only = True, mode = 'auto'),
		EarlyStopping(monitor=monitor, patience = 5),
		ReduceLROnPlateau(monitor=monitor, factor=0.2, patience = 2, min_lr=1e-8, mode='auto'),
		CSVLogger(LOG_FPATH, separator= ' ', append = True),
		]
	return callbacks

def class_weight_dict(labels_dict, mu=0.15):
	# labels_dict : {ind_label: count_label}
	# mu : parameter to tune
	total = np.sum(list(labels_dict.values()))
	keys = list(labels_dict.keys())
	class_weight = dict()
	for key in keys:
		score = np.log(mu*total/float(labels_dict[key]))
		class_weight[key] = score if score > 1.0 else 1.0
	return class_weight

def class_report(X, y, model, outfile = "class_report"):
	output_labels = {0: 'publisher', 1: 'fact', 2: 'bias'}
	y_pred = model.predict(X)
	for k,(_y, _y_pred) in enumerate(zip(y, y_pred)):
		report = classification_report(_y, _y_pred.round(), output_dict=True)
		report = pd.DataFrame(report).transpose()
		report.to_csv(outfile + ".{}.csv".format(output_labels[k]))

def doc_vectors(model, tagged_docs, label_index=0, reinfer_train=False, infer_steps=5, infer_alpha=None, min_words = 1):
	docvals = tagged_docs.values
	docvals = [doc for doc in docvals if len(doc.words) >= min_words]
	print("Total documents with length >= {}: {}".format(min_words,len(docvals)))

	def _get(doc):
		if label_index in doc.tags:
			return (doc.tags[0], doc.tags[1], doc.tags[2], model.docvecs[doc.tags[label_index]])
		else:
			return (doc.tags[0], doc.tags[1], doc.tags[2], model.infer_vector(doc.words, steps=infer_steps))

	y1, y2, y3, x = zip(*[_get(doc) for doc in docvals])
	return np.array(x), y1, y2, y3

def load_cached_vectors(df, model, label_index, min_words = 10, num_classes = 938, cache_loc = None, fit_encoder = False, infer_steps = None, infer_alpha = None, reinfer_train = True):
    # If vectors already saved, load them --> infer step is slow.
    if os.path.isfile(cache_loc):
        with open(cache_loc, "rb") as fid:
            X, y1, y2, y3 = pickle.load(fid)
    else:
        # Tag the docs
        tagged = df.apply(lambda r: TaggedDocument(words=r['docu_text'],
        					   tags=[r.publisher, r.fact_score, r.bias_score]),
        					   axis=1)

        # Load the document vectors (using publisher label)
        X, y1, y2, y3 = doc_vectors(model, tagged,
        							min_words = min_words,
        							label_index = label_index,
        							infer_steps = infer_steps,
        							infer_alpha = infer_alpha,
        							reinfer_train = reinfer_train)

        label_encoder1 = LabelEncoder()
        label_encoder1.fit(y1)
        label_encoder2 = LabelEncoder()
        label_encoder2.fit(y2)
        label_encoder3 = LabelEncoder()
        label_encoder3.fit(y3)

        y1 = np_utils.to_categorical((label_encoder1.transform(list(y1))),
        								   num_classes = num_classes)
        y2 = np_utils.to_categorical((label_encoder2.transform(list(y2))),
        								   num_classes = 3)
        y3 = np_utils.to_categorical((label_encoder3.transform(list(y3))),
        								   num_classes = 7)

        with open(cache_loc, "wb") as fid:
        	pickle.dump([X, y1, y2, y3],fid)

    return X, y1, y2, y3

class FakeNewsModel:

    def __init__(self, word_vector_dim, N_classes, N_fact = 3, N_bias = 7):
        self.word_vector_dim = word_vector_dim
        self.N_classes = N_classes
        self.N_fact = N_fact
        self.N_bias = N_bias

    def shared_block(self, inputs):
        x = Dense(1000, input_dim = self.word_vector_dim, kernel_initializer="normal", kernel_regularizer=l2(0.0))(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)
        return x

    def publisher(self, inputs):
    	x = Dense(self.N_classes, kernel_initializer="normal", activation="softmax", name="pub")(inputs)
    	return x

    def fact(self, inputs):
    	x = Dense(self.N_fact, kernel_initializer="normal", activation="softmax", name = "fact")(inputs)
    	return x

    def bias(self, inputs):
    	x = Dense(self.N_bias, kernel_initializer="normal", activation="softmax", name = "bias")(inputs)
    	return x

    def score(self, inputs):
    	x = Dense(1, kernel_initializer="normal", activation="linear", name = "score")(inputs)
    	return x

    def model(self):
    	_input = Input(shape=(self.word_vector_dim,))
    	x = self.shared_block(_input)
    	pub_out = self.publisher(x)
    	fact_out = self.fact(x)
    	bias_out = self.bias(x)
    	model = Model(inputs = _input, outputs = [pub_out, fact_out, bias_out])

    	score_out = self.score(x)
    	score_model = Model(inputs = _input, outputs = score_out)
    	return model, score_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_epoch', type = int, default = 100,
    					help = 'Number of epochs to train for, default 100')
    parser.add_argument('--batch_size', type = int, default = 16,
    					help = 'Batch size, default 16')
    parser.add_argument('--hidden_dim', type = int, default = 1000,
    					help = 'Batch size, default 1000')
    parser.add_argument('--min_doc_size', type = int, default = 10,
    					help = 'Minimum length of title, default 10')
    parser.add_argument('--label_index', type = int, default = 0,
    					help = '0: publishers, 1: fact, 2: bias. Default: 0')
    parser.add_argument('--retrain', type = bool, default = False,
    	                help = 'Retrain with loaded weights, default False')
    parser.add_argument('--l1', type = float, default = 0,
    					help = 'l1 regularization parameter for each Dense layer, default 0')
    parser.add_argument('--l2', type = float, default = 0,
    					help = 'l2 regularization parameter for each Dense layer, default 0')
    parser.add_argument('--lr', type = float, default = 0.001,
    					help = 'Learning rate, default 1e-3')
    parser.add_argument('--docvec_size', type = int, default = 1000,
    					help = 'Document vector size, default 1000')
    parser.add_argument('--infer_steps', type = int, default = 20,
    					help = 'Infer steps for document vectors, default 20')
    parser.add_argument('--save_doc2vec_loc', type = str, default = 'data', help = 'Save processed training data to csv')
    parser.add_argument('--save_classifier_loc', type = str, default = 'models', help = 'Save processed training data to csv')
    args = parser.parse_args()

    N_epochs		= int(args.nb_epoch)
    batch_size		= int(args.batch_size)
    hidden_dim		= int(args.hidden_dim)
    vec_size		= int(args.docvec_size)
    infer_steps     = int(args.infer_steps)
    label_index     = int(args.label_index)
    min_doc_size 	= int(args.min_doc_size)
    data_save_dir	= str(args.save_doc2vec_loc)
    model_save_dir	= str(args.save_classifier_loc)
    lr 				= float(args.lr)

    # Load doc2vec csv / pkl preprocessed data, used to get vectors
    doc2vec_training_csv = '{}/Doc2vecTrainingData.csv'.format(data_save_dir)
    doc2vec_preprocessed = '{}/Doc2vecTrainingDataProcessed.pkl'.format(data_save_dir)
    df = load_data(doc2vec_training_csv = doc2vec_training_csv,
    				doc2vec_preprocessed = doc2vec_preprocessed)

    # Misc. filepaths for training and results data
    LOG_FPATH = '{}/classifier_training.txt'.format(model_save_dir)
    DOC2VEC_MODEL = "{}/doc2vec/doc2vec-VecSize-{}_MinDoc-{}_Window-300.model".format(model_save_dir,vec_size,min_doc_size)
    WEIGHTS_FPATH = "{}/classifier/model.h5".format(model_save_dir)
    CACHE_VECTOR_LOC = "{}/doc2vec/cached_vectors_bulk.pkl".format(model_save_dir)
    PLOT_DATA_LOC = "{}/plot_data.pkl".format(data_save_dir)

    # Load trained doc2vec model
    model = Doc2Vec.load(DOC2VEC_MODEL)

    # Number of unique labels
    N_vocab = len(model.wv.vocab)
    N_classes = len(model.docvecs)

    print("Training classifier to predict publishers given words")
    print("... loading doc2vec model")
    print("... length of vocabulary:", N_vocab)
    print("... number of doc-labels:", N_classes)

    # Get mapping from publisher to id
    print("... loading label encoder for the full data-set")
    X, y1, y2, y3 = load_cached_vectors(df, model, label_index,
									   min_words = min_doc_size,
									   num_classes = N_classes,
									   cache_loc = CACHE_VECTOR_LOC)

    # Set up class-imbalance weights using scheme
    freq1 = CountFrequency(np.argmax(y1, axis=1))
    freq2 = CountFrequency(np.argmax(y2, axis=1))
    freq3 = CountFrequency(np.argmax(y3, axis=1))
    class_weights = {'pub': class_weight_dict(freq1, mu = 0.2),
                     'fact': class_weight_dict(freq2, mu = 0.2),
                     'bias': class_weight_dict(freq3, mu = 0.2)}

    # Split data (default set to be 70/30, ignoring 30 --> 15-15 plot)
    print("... loading training data")
    y_train = []
    y_test  = []
    for y in [y1,y2,y3]:
        X_train, X_test, y_tr, y_te = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)
        y_train.append(y_tr)
        y_test.append(y_te)

    ###########################################################

    print("... fitting the classifier model")

    # Load neural classifier model
    classifier, score_model = FakeNewsModel(vec_size, N_classes).model()
    classifier.summary()

    # Load callbacks, optimizer, loss details
    callbacks = get_callbacks(WEIGHTS_FPATH, LOG_FPATH, "val_fact_loss")

    # Use adam optimer, clip
    adam = Adam(lr=lr, clipnorm=1.0)

    # Multi-head loss. Keras adds by default
    losses = {"pub": 'categorical_crossentropy',
    		  "fact": 'categorical_crossentropy',
    		  "bias": 'categorical_crossentropy'}

    lossWeights = {"pub": 1.0,
                   "fact": np.log(N_classes/3),
                   "bias": np.log(N_classes/7)}

    metrics = ['accuracy']#, top3_acc, top5_acc, top10_acc]
    classifier.compile(loss=losses, loss_weights=lossWeights,
                        optimizer=adam, metrics=metrics)

    # Train the model
    estimator = classifier.fit(X_train, y_train,
    							validation_data = (X_test, y_test),
    					 		epochs = N_epochs,
    					 		callbacks = callbacks,
    					 		shuffle = True,
    					 		batch_size = batch_size,
    					 		verbose = 2,
    					 		class_weight = class_weights,
    					 		)

    print("... publisher: train accuracy: %.2f%% / val accuracy: %.2f%%" % (100*estimator.history['pub_acc'][-1], 100*estimator.history['val_pub_acc'][-1]))
    print("... factual: train accuracy: %.2f%% / val accuracy: %.2f%%" % (100*estimator.history['fact_acc'][-1], 100*estimator.history['val_fact_acc'][-1]))
    print("... bias: train accuracy: %.2f%% / val accuracy: %.2f%%" % (100*estimator.history['bias_acc'][-1], 100*estimator.history['val_bias_acc'][-1]))

    # Breakdown of class accuracies, etc.
    class_report(X_train, y_train, classifier,
                outfile = "results/class_report_train")
    class_report(X_test, y_test, classifier,
                outfile = "results/class_report")

    # Save model weights to h5
    classifier.save_weights(WEIGHTS_FPATH, overwrite = True)

    # Save model + weights to pkl for faster loading with app
    with open(WEIGHTS_FPATH + '.pkl', "wb") as fid:
        pickle.dump(classifier, fid)

    # Save results for making plots, reports, etc.
    with open(PLOT_DATA_LOC, "wb") as fid:
    	pickle.dump([estimator, N_classes, X_train, y_train, X_test, y_test],fid)
