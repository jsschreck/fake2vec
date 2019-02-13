from sklearn.model_selection import train_test_split
from fake2vec.utils.text_processor import WordsContainer

import pandas as pd, numpy as np, pickle
import sys, re, os, multiprocessing, random, time, argparse

import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import utils

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def fact_to_token(value):
	values = ['HIGH', 'MIXED', 'LOW']
	index = values.index(value)
	return index

def bias_to_token(value):
	values = ['extreme-left', 'left', 'left-center', 'center', 'right-center', 'right', 'extreme-right']
	index = values.index(value)
	return index

def load_data(doc2vec_training_csv = 'data/Doc2vecTrainingData.csv', doc2vec_preprocessed = 'data/Doc2vecTrainingData.pkl'):

	if os.path.isfile(doc2vec_training_csv):
		df = pd.read_csv(doc2vec_training_csv, index_col=False)
	else:
		# Load the training articles + labels
		data = WordsContainer()
		data.load(doc2vec_preprocessed)
		myVocab = data.words
		myArticles = data.articles

		companies = myArticles.keys()
		N_classes = len(companies)
		company_to_token = {company: k for k,company in zip(range(len(companies)),companies)}
		token_to_company = {k: company for k,company in zip(range(len(companies)),companies)}

		with open(doc2vec_training_csv, 'w') as fid:
			fid.write('docu_text,publisher,publisher_token,fact_score,fact_token,bias_score,bias_token\n')

		docs_dict = {'docu_text': [], 'publisher': [], 'publisher_token': [], 'fact_score': [], 'fact_token': [], 'bias_score': [], 'bias_token': []}
		for (company,metadata) in list(myArticles.items()):

			tokens = metadata['tokens']
			articles = metadata['articles']
			fact_score = metadata['fact']
			bias_score = metadata['bias']

			company_token = company_to_token[company]
			fact_token = fact_to_token(fact_score)
			bias_token = bias_to_token(bias_score)

			for (token,article) in zip(tokens,articles):

				title = article['title']
				summary = article['summary']
				keywords = article['keywords']
				sentences = article['sentences']

				sentences = [item for sublist in sentences for item in sublist]
				article_document = title + keywords + summary + sentences
				article_string = " ".join([word for word in article_document if word.isalpha()])

				if article_string == "" or article_string == ' ':
					continue

				docs_dict['docu_text'].append(article_string)
				docs_dict['publisher'].append(company)
				docs_dict['publisher_token'].append(company_token)
				docs_dict['fact_score'].append(fact_score)
				docs_dict['fact_token'].append(fact_token)
				docs_dict['bias_score'].append(bias_score)
				docs_dict['bias_token'].append(bias_token)

		# Load data into pandas
		df = pd.DataFrame.from_dict(docs_dict)
		df.to_csv(doc2vec_training_csv, index=False)

	# Word count
	df['docu_text'] = df['docu_text'].apply(lambda x: x.split(" "))
	total_words = df['docu_text'].apply(lambda x: len(x)).sum()

	# Print number of documents and total word count
	print("Number of text docs / fields:", df.shape)
	print("Total word count:", total_words)

	# Shuffle the rows
	df = utils.shuffle(df)

	return df

def top_n_accuracy(preds, truths, n):
	best_n = np.argsort(preds, axis=1)[:,-n:]
	ts = np.argmax(truths, axis=1)
	successes = 0
	for i in range(ts.shape[0]):
		if ts[i] in best_n[i,:]:
			successes += 1
	return float(successes)/ts.shape[0]

def get_vectors(model, tagged_docs, label_index=1, reinfer_train=False, infer_steps=5, infer_alpha=None, min_words = 1):

	docvals = tagged_docs.values
	docvals = [doc for doc in docvals if len(doc.words) >= min_words]
	print("... total documents with length >= {}: {}".format(min_words,len(docvals)))

	if reinfer_train:
		targets, regressors = zip(*[(doc.tags[label_index], model.infer_vector(doc.words, steps=infer_steps)) for doc in docvals])

	else:
		def _get(doc):
			if label_index in doc.tags:
				return (doc.tags[label_index], model.docvecs[doc.tags[label_index]])
			else:
				return (doc.tags[label_index], model.infer_vector(doc.words, steps=infer_steps))

		targets, regressors = zip(*[_get(doc) for doc in docvals])

	return targets, regressors

def LogisticClassifier(model, train, test, label_index=0, infer_steps=None, infer_alpha=None, cores = 1, min_words = 1):
	y_train, X_train = get_vectors(model, train,
									label_index = label_index,
									infer_steps = infer_steps,
									infer_alpha = infer_alpha,
									min_words = min_words,
									reinfer_train = True)

	y_test, X_test = get_vectors(model, test,
									label_index = label_index,
									infer_steps = infer_steps,
									infer_alpha = infer_alpha,
									min_words = min_words,
									reinfer_train = True)

	predictor = LogisticRegression(solver='lbfgs',
								   multi_class='multinomial',
								   n_jobs=cores,
								   class_weight='balanced',
								   C=1e9,
								   random_state=0,
								   max_iter=100) #maxiter=100

	predictor.fit(X_train, y_train)

	y_pred_train = predictor.predict(X_train)
	y_pred_test = predictor.predict(X_test)

	train_acc = accuracy_score(y_train, y_pred_train)
	test_acc = accuracy_score(y_test, y_pred_test)

	print('... Train: accuracy {}'.format(train_acc))
	print('... Test: accuracy {}'.format(test_acc))

	report_train = classification_report(y_train, y_pred_train, output_dict=True)
	report_test = classification_report(y_test, y_pred_test, output_dict=True)

	return report_train, report_test, predictor

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--nb_epoch', type = int, default = 30,
						help = 'Number of epochs to train for, default 1000')
	parser.add_argument('--docvec_size', type = int, default = 300,
						help = 'Document vector size, default 300')
	parser.add_argument('--window', type = int, default = 300,
						help = 'Doc2vec training window, default 300')
	parser.add_argument('--min_word_count', type = int, default = 10,
						help = 'Minimum number of words in document, default 10')
	parser.add_argument('--cores', type = int, default = multiprocessing.cpu_count(),
						help = 'Default is number of cores available.')
	parser.add_argument('--save_training_doc2vec', type = str, default = 'data',
						help = 'Save processed training data to csv')
	args = parser.parse_args()

	N_epochs		= int(args.nb_epoch)
	vec_size		= int(args.docvec_size)
	window			= int(args.window)
	min_count  		= int(args.min_word_count)
	save_dir		= str(args.save_training_doc2vec)
	cores 			= int(args.cores)

	model_details_tag = "VecSize-{}_MinDoc-{}_Window-{}".format(vec_size,min_count,window)

	# Load file paths
	doc2vec_training_csv = '{}/Doc2vecTrainingData.csv'.format(save_dir)
	doc2vec_preprocessed = '{}/Doc2vecTrainingDataProcessed.pkl'.format(save_dir)

	# Save file paths
	doc2vec_model_loc = "models/production/doc2vec-{}.model".format(model_details_tag)
	classifier_model_loc = "models/production/logistic_classifier-{}.pkl".format(model_details_tag)
	classifier_report_csv = "reports/production/logistic_classifier_report-{}.csv".format(model_details_tag)
	check_classifier_acc_every = 1

	assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"
	assert os.path.isfile(doc2vec_preprocessed), "Load preprocessed file. Otherwise, python utils/text_processor.py"

	# Load data into pandas dataframe
	df = load_data(doc2vec_training_csv = doc2vec_training_csv,
					doc2vec_preprocessed = doc2vec_preprocessed)

	# Split into train/test for training doc2vec model.
	train, test = train_test_split(df, test_size=0.3, random_state=42)

	# tags - [publisher, fact, bias]
	train_tagged = train.apply(lambda r: TaggedDocument(words=r['docu_text'],
							   tags=[r.publisher, r.fact_score, r.bias_score]),
							   axis=1)
	test_tagged  = test.apply(lambda r: TaggedDocument(words=r['docu_text'],
	    					   tags=[r.publisher, r.fact_score, r.bias_score]),
	    					   axis=1)

	# Initialize Doc2Vec model
	alpha = 0.025
	min_alpha = 1e-4
	alpha_delta = (alpha - min_alpha) / (N_epochs - 1)

	model = Doc2Vec(vector_size = vec_size,
					window = window,
	                alpha = alpha,
	                min_alpha = min_alpha,
	                min_count = min_count,
	                hs = 1,
	                #negative = 5,
	                sample = 0,
	                workers = cores,
	                #dbow_words = 1,
	                dm = 0)
	model.build_vocab(train_tagged.values)

	# Train
	print("Training for a maximum of {} epochs".format(N_epochs))
	for epoch in range(N_epochs):
		print('Epoch {} -'.format(epoch))

		t0 = time.time()
		model.train(utils.shuffle(train_tagged.values),
					total_examples = len(train_tagged.values),
					epochs = 1)

		# decrease the learning rate
		model.alpha -= 0.002
		#model.alpha -= alpha_delta

		# fix the learning rate, no decay
		model.min_alpha = model.alpha

		# Check acc. on LogsiticClassifier
		if (epoch + 1) % check_classifier_acc_every == 0:
			print("... evaluating %s" % model)

			label = 0  # publishers
			report_train, report_test, classifier = LogisticClassifier(model,
																train_tagged,
																test_tagged,
																cores = cores,
																min_words = min_count,
																label_index = label)
			report_train = pd.DataFrame(report_train).transpose()
			report_train.to_csv(classifier_report_csv)

			with open(classifier_model_loc + "_{}".format(epoch), "wb") as fid:
				pickle.dump(classifier,fid)

			print("... saving model weights")
			model.save(doc2vec_model_loc + "_{}".format(epoch))

		print('... it took {:.2f}s to complete the epoch'.format(time.time()-t0))

	'''
	# Build pyTorch for deep-classifier ...
	batch_size = 32
	label_index = 0

	train, test = train_test_split(df, test_size=0.2)
	test, val = train_test_split(test, test_size=0.5)

	train_tagged = train.apply(lambda r: TaggedDocument(words=r['docu_text'],
							   tags=[r.publisher, r.fact_score, r.bias_score]),
							   axis=1)
	test_tagged  = test.apply(lambda r: TaggedDocument(words=r['docu_text'],
	    					   tags=[r.publisher, r.fact_score, r.bias_score]),
	    					   axis=1)
	val_tagged  = val.apply(lambda r: TaggedDocument(words=r['docu_text'],
	    					   tags=[r.publisher, r.fact_score, r.bias_score]),
	    					   axis=1)

	y_train, X_train = get_vectors(model, train_tagged,
									label_index = label_index,
									infer_steps = infer_steps,
									infer_alpha = infer_alpha,
									reinfer_train = False)

	y_test, X_test = get_vectors(model, test_tagged,
									label_index = label_index,
									infer_steps = infer_steps,
									infer_alpha = infer_alpha,
									reinfer_train = False)

	y_val, X_val = get_vectors(model, val_tagged,
									label_index = label_index,
									infer_steps = infer_steps,
									infer_alpha = infer_alpha,
									reinfer_train = False)

	from sklearn.preprocessing import LabelEncoder
	label_train = LabelEncoder()
	label_train.fit(y_train)
	y_train = np_utils.to_categorical((label_encoder.transform(y_train)))

	# Train-data generator
	data_train = TorchDataGenerator(X_train, y_train)
	dataGen_train = DataLoader(data_train,
                               batch_size = batch_size,
                               shuffle = True,
                               num_workers = 4)

	# Test-data generator
	data_test = TorchDataGenerator(X_test, y_test)
	dataGen_test = DataLoader(data_test,
                               batch_size = batch_size,
                               shuffle = False,
                               num_workers = 4)

	# Test-data generator
	data_val = TorchDataGenerator(X_val, y_val)
	dataGen_val = DataLoader(data_val,
                               batch_size = batch_size,
                               shuffle = False,
                               num_workers = 4)

	# Categorical cross entropy
	loss_func = nn.CrossEntropyLoss()

	# Load optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr = lr)
	#optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)

	'''
