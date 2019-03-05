from sklearn.model_selection import train_test_split
from fake2vec.utils.text_processor import WordsContainer
from fake2vec.utils.model import doc2vec

import pandas as pd, numpy as np, pickle
import sys, re, os, multiprocessing, random, time, argparse

import gensim
#from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import utils

from keras.utils import np_utils

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def fact_to_token(value):
    values = ['HIGH', 'MIXED', 'LOW']
    index = values.index(value)
    return index


def bias_to_token(value):
    values = [
        'extreme-left', 'left', 'left-center', 'center', 'right-center',
        'right', 'extreme-right'
    ]
    index = values.index(value)
    return index


def load_data(doc2vec_training_csv='data/Doc2vecTrainingData.csv',
              doc2vec_preprocessed='data/Doc2vecTrainingData.pkl'):

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
        company_to_token = {
            company: k
            for k, company in zip(range(len(companies)), companies)
        }
        token_to_company = {
            k: company
            for k, company in zip(range(len(companies)), companies)
        }

        with open(doc2vec_training_csv, 'w') as fid:
            fid.write(
                'docu_text,publisher,publisher_token,fact_score,fact_token,bias_score,bias_token\n'
            )

        docs_dict = {
            'docu_text': [],
            'publisher': [],
            'publisher_token': [],
            'fact_score': [],
            'fact_token': [],
            'bias_score': [],
            'bias_token': []
        }
        for (company, metadata) in list(myArticles.items()):

            tokens = metadata['tokens']
            articles = metadata['articles']
            fact_score = metadata['fact']
            bias_score = metadata['bias']

            company_token = company_to_token[company]
            fact_token = fact_to_token(fact_score)
            bias_token = bias_to_token(bias_score)

            for (token, article) in zip(tokens, articles):

                title = article['title']
                summary = article['summary']
                keywords = article['keywords']
                sentences = article['sentences']

                sentences = [item for sublist in sentences for item in sublist]
                article_document = title + keywords + summary + sentences
                article_string = " ".join(
                    [word for word in article_document if word.isalpha()])

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
    df = utils.shuffle(df, random_state=123)
    return df


def top_n_accuracy(preds, truths, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    ts = np.argmax(truths, axis=1)
    successes = 0
    for i in range(ts.shape[0]):
        if ts[i] in best_n[i, :]:
            successes += 1
    return float(successes) / ts.shape[0]


def LogisticClassifier(Model,
                       train,
                       test,
                       le,
                       label_index=0,
                       infer_steps=0,
                       infer_alpha=False,
                       min_count=10,
                       num_classes=938,
                       cores=1):

    X_train, y_train, _, _ = Model.doc_vectors(
        train,
        label_index=label_index,
        infer_steps=infer_steps,
        infer_alpha=infer_alpha,
        min_words=min_count,
        reinfer_train=False)
    X_test, y_test, _, _ = Model.doc_vectors(
        test,
        label_index=label_index,
        infer_steps=infer_steps,
        infer_alpha=infer_alpha,
        min_words=min_count,
        reinfer_train=False)

    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    predictor = LogisticRegression(
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=cores,
        class_weight='balanced',
        C=1e9,
        random_state=0,
        max_iter=100)  #maxiter=100

    predictor.fit(X_train, y_train)

    y_pred_train = predictor.predict(X_train)
    y_pred_test = predictor.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print('... Train: accuracy {}'.format(train_acc))
    print('... Test: accuracy {}'.format(test_acc))

    report_train = classification_report(
        y_train, y_pred_train, output_dict=True)
    report_test = classification_report(y_test, y_pred_test, output_dict=True)

    return report_train, report_test, predictor


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nb_epoch',
        type=int,
        default=20,
        help='Number of epochs to train for, default 30')
    parser.add_argument(
        '--docvec_size',
        type=int,
        default=1000,
        help='Document vector size, default 1000')
    parser.add_argument(
        '--window',
        type=int,
        default=300,
        help='Doc2vec training window, default 300')
    parser.add_argument(
        '--min_word_count',
        type=int,
        default=10,
        help='Minimum number of words in document, default 10')
    parser.add_argument(
        '--cores',
        type=int,
        default=multiprocessing.cpu_count(),
        help='Default is number of cores available.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./',
        help='Save processed training data to csv')
    args = parser.parse_args()

    N_epochs = int(args.nb_epoch)
    vec_size = int(args.docvec_size)
    window = int(args.window)
    min_count = int(args.min_word_count)
    save_dir = str(args.save_dir)
    cores = int(args.cores)

    model_details_tag = "VecSize-{}_MinDoc-{}_Window-{}".format(
        vec_size, min_count, window)
    # Need to check if directories exist.
    data_path = os.path.join(save_dir, 'data')
    model_path = os.path.join(save_dir, 'models/doc2vec')
    #results_path = os.path.join(save_dir, 'results')
    for data_dir in [save_dir, model_path]:
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

    # Load file paths
    doc2vec_training_csv = '{}/Doc2vecTrainingData.csv'.format(data_path)
    doc2vec_preprocessed = '{}/Doc2vecTrainingDataProcessed.pkl'.format(
        data_path)

    # Save file paths
    label_encoders = "{}/label_encoders.pkl".format(data_path)
    doc2vec_model_loc = "{}/doc2vec-{}.model".format(model_path,
                                                     model_details_tag)
    classifier_model_loc = "{}/logistic_classifier-{}.pkl".format(
        model_path, model_details_tag)
    classifier_report_csv = "{}/logistic_classifier_report-{}.csv".format(
        model_path, model_details_tag)
    check_classifier_acc_every = 1

    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully \
													 slow otherwise"

    assert os.path.isfile(doc2vec_preprocessed), "Load preprocessed file. \
									Otherwise, python utils/text_processor.py"

    # Load data into pandas dataframe
    df = load_data(
        doc2vec_training_csv=doc2vec_training_csv,
        doc2vec_preprocessed=doc2vec_preprocessed)

    # Create the label encoders and save. These are used later to evaluate model
    le1 = LabelEncoder()
    le1.fit(df['publisher'])
    le2 = LabelEncoder()
    le2.fit(df['fact_score'])
    le3 = LabelEncoder()
    le3.fit(df['bias_score'])

    encoders = [le1, le2, le3]
    label_types = ['publisher', 'fact', 'bias']

    # Count the max number of labels per label-type
    N_pub_labels = max(le1.transform(df['publisher'])) + 1
    N_fact_labels = max(le2.transform(df['fact_score'])) + 1
    N_bias_labels = max(le3.transform(df['bias_score'])) + 1
    classes_counter = [N_pub_labels, N_fact_labels, N_bias_labels]

    # Save the label encoders.
    # Used to train classifier and in deployed model
    with open(label_encoders, "wb") as fid:
        pickle.dump([encoders, classes_counter], fid)

    # Split into train/test for training doc2vec model.
    label_index = 0

    train, test = train_test_split(df, test_size=0.3, random_state=42)

    # tags - [publisher, fact, bias]
    train_tagged = train.apply(
        lambda r: TaggedDocument(
            words=r['docu_text'],
            tags=[r.publisher, r.fact_score, r.bias_score]),
        axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(
            words=r['docu_text'],
            tags=[r.publisher, r.fact_score, r.bias_score]),
        axis=1)

    # Load custom doc2vec model class with some default values
    alpha = 0.025
    min_alpha = 1e-4
    alpha_delta = (alpha - min_alpha) / (N_epochs - 1)

    Model = doc2vec(
        vec_size=vec_size,
        window=window,
        alpha=alpha,
        min_alpha=min_alpha,
        min_count=min_count,
        cores=cores)
    model = Model.model
    model.build_vocab(train_tagged.values)

    # Train the model
    print("Training for a maximum of {} epochs".format(N_epochs))
    print("Using classifier label: {}".format(label_types[label_index]))

    for epoch in range(N_epochs):
        print('Epoch {} -'.format(epoch))

        t0 = time.time()
        model.train(
            utils.shuffle(train_tagged.values),
            total_examples=len(train_tagged.values),
            epochs=1)

        # decrease the learning rate
        model.alpha -= 0.002
        #model.alpha -= alpha_delta

        # fix the learning rate, no decay
        model.min_alpha = model.alpha

        # Check acc. on LogsiticClassifier
        if (epoch + 1) % check_classifier_acc_every == 0:
            print("... evaluating %s" % model)

            report_train, report_test, classifier = LogisticClassifier(
                Model,
                train_tagged,
                test_tagged,
                encoders[label_index],
                label_index=label_index,
                infer_steps=0,
                num_classes=N_pub_labels,
                min_count=min_count,
                cores=cores)
            report_train = pd.DataFrame(report_train).transpose()
            report_train.to_csv(classifier_report_csv)

            with open(classifier_model_loc + "_{}".format(epoch), "wb") as fid:
                pickle.dump(classifier, fid)

            print("... saving model weights")
            # Save using doc2vec save method
            model.save(doc2vec_model_loc + "_{}".format(epoch))
            # Dump loaded file to pkl
            with open(doc2vec_model_loc + "_{}.pkl".format(epoch),
                      "wb") as fid:
                pickle.dump(model, fid)

        print('... it took {:.2f}s to complete the epoch'.format(time.time() -
                                                                 t0))