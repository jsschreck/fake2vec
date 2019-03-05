from sklearn.model_selection import train_test_split
from fake2vec.utils.text_processor import WordsContainer
from fake2vec.utils.train_doc2vec import load_data
from fake2vec.utils.model import doc2vec, KerasClassifier

import numpy as np
import pandas as pd
import sys, re, os, argparse, pickle, functools, collections, warnings

from gensim.models.doc2vec import TaggedDocument

import keras
from keras.utils import np_utils
from keras.regularizers import l1, l2
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn import utils

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
        ModelCheckpoint(
            WEIGHTS_FPATH, monitor=monitor, save_best_only=True, mode='auto'),
        EarlyStopping(monitor=monitor, patience=5),
        ReduceLROnPlateau(
            monitor=monitor, factor=0.2, patience=2, min_lr=1e-8, mode='auto'),
        CSVLogger(LOG_FPATH, separator=' ', append=True),
    ]
    return callbacks


def class_weight_dict(labels_dict, mu=0.15):
    # labels_dict : {ind_label: count_label}
    # mu : parameter to tune
    total = np.sum(list(labels_dict.values()))
    keys = list(labels_dict.keys())
    class_weight = dict()
    for key in keys:
        score = np.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight


def class_report(X, y, model, outfile="class_report"):
    output_labels = {0: 'publisher', 1: 'fact', 2: 'bias'}
    y_pred = model.predict(X)
    for k, (_y, _y_pred) in enumerate(zip(y, y_pred)):
        report = classification_report(_y, _y_pred.round(), output_dict=True)
        report = pd.DataFrame(report).transpose()
        report.to_csv(outfile + ".{}.csv".format(output_labels[k]))


def load_cached_vectors(df,
                        model,
                        label_index,
                        encoders,
                        min_words=10,
                        num_classes=938,
                        cache_loc=None,
                        fit_encoder=False,
                        infer_steps=None,
                        infer_alpha=None,
                        reinfer_train=True):

    # If vectors already saved, load them --> infer step is slow.
    if os.path.isfile(cache_loc):
        with open(cache_loc, "rb") as fid:
            X, y1, y2, y3 = pickle.load(fid)
    else:
        # Tag the docs
        tagged = df.apply(
            lambda r: TaggedDocument(
                words=r['docu_text'],
                tags=[r.publisher, r.fact_score, r.bias_score]),
            axis=1)

        # Load the document vectors (using publisher label)
        X, y1, y2, y3 = model.doc_vectors(
            tagged,
            min_words=min_words,
            label_index=label_index,
            infer_steps=infer_steps,
            infer_alpha=infer_alpha,
            reinfer_train=reinfer_train)

        y1 = encoders[0].transform(y1)
        y2 = encoders[1].transform(y2)
        y3 = encoders[2].transform(y3)

        y1 = np_utils.to_categorical(y1, num_classes=num_classes)
        y2 = np_utils.to_categorical(y2, num_classes=3)
        y3 = np_utils.to_categorical(y3, num_classes=7)

        with open(cache_loc, "wb") as fid:
            pickle.dump([X, y1, y2, y3], fid)

    return X, y1, y2, y3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nb_epoch',
        type=int,
        default=100,
        help='Number of epochs to train for, default 100')
    parser.add_argument(
        '--batch_size', type=int, default=16, help='Batch size, default 16')
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=1000,
        help='Batch size, default 1000')
    parser.add_argument(
        '--min_doc_size',
        type=int,
        default=10,
        help='Minimum length of title, default 10')
    parser.add_argument(
        '--label_index',
        type=int,
        default=0,
        help='0: publishers, 1: fact, 2: bias. Default: 0')
    parser.add_argument(
        '--retrain',
        type=bool,
        default=False,
        help='Retrain with loaded weights, default False')
    parser.add_argument(
        '--l1',
        type=float,
        default=0,
        help='l1 regularization parameter for each Dense layer, default 0')
    parser.add_argument(
        '--l2',
        type=float,
        default=0,
        help='l2 regularization parameter for each Dense layer, default 0')
    parser.add_argument(
        '--lr', type=float, default=0.001, help='Learning rate, default 1e-3')
    parser.add_argument(
        '--docvec_size',
        type=int,
        default=1000,
        help='Document vector size, default 1000')
    parser.add_argument(
        '--infer_steps',
        type=int,
        default=20,
        help='Infer steps for document vectors, default 20')
    parser.add_argument(
        '--save_loc', type=str, default='./', help='Save output to dir')
    args = parser.parse_args()

    N_epochs = int(args.nb_epoch)
    batch_size = int(args.batch_size)
    hidden_dim = int(args.hidden_dim)
    vec_size = int(args.docvec_size)
    infer_steps = int(args.infer_steps)
    label_index = int(args.label_index)
    min_doc_size = int(args.min_doc_size)
    data_save_dir = str(args.save_loc)
    lr = float(args.lr)

    # Load doc2vec csv / pkl preprocessed data, used to get vectors
    joined_dir = os.path.join(data_save_dir, 'data')
    if not os.path.isdir(joined_dir):
        os.makedirs(joined_dir)
    doc2vec_training_csv = '{}/Doc2vecTrainingData.csv'.format(joined_dir)
    doc2vec_preprocessed = '{}/Doc2vecTrainingDataProcessed.pkl'.format(
        joined_dir)
    label_encoders = "{}/label_encoders.pkl".format(joined_dir)
    PLOT_DATA_LOC = "{}/plot_data.pkl".format(joined_dir)
    df = load_data(
        doc2vec_training_csv=doc2vec_training_csv,
        doc2vec_preprocessed=doc2vec_preprocessed)

    # Misc. filepaths for training and results data
    joined_dir = os.path.join(data_save_dir, 'models')
    if not os.path.isdir(joined_dir):
        os.makedirs(data_dir)
        os.makedirs(data_dir + '/doc2vec')
        os.makedirs(data_dir + '/classifier')

    DOC2VEC_MODEL = "{}/doc2vec/doc2vec-VecSize-{}_MinDoc-{}_Window-300.model".format(
        joined_dir, vec_size, min_doc_size)
    CACHE_VECTOR_LOC = "{}/doc2vec/cached_vectors_bulk.pkl".format(joined_dir)
    LOG_FPATH = '{}/classifier/classifier_training.txt'.format(joined_dir)
    WEIGHTS_FPATH = "{}/classifier/model.h5".format(joined_dir)

    # Load label encoders
    with open(label_encoders, "rb") as fid:
        encoders, classes_counter = pickle.load(fid)
        N_pubs, N_fact, N_bias = classes_counter

    # Load trained doc2vec model
    model = doc2vec(DOC2VEC_MODEL)

    # Number of unique labels
    N_vocab = len(model.model.wv.vocab)

    print("Training classifier to predict publishers given words")
    print("... loading doc2vec model")
    print("... length of vocabulary:", N_vocab)
    print("... number of pub labels:", N_pubs)
    print("... number of fact labels:", N_fact)
    print("... number of bias labels:", N_bias)

    # Get mapping from publisher to id
    X, y1, y2, y3 = load_cached_vectors(
        df,
        model,
        label_index,
        encoders,
        min_words=min_doc_size,
        num_classes=N_pubs,
        infer_steps=infer_steps,
        cache_loc=CACHE_VECTOR_LOC)

    # Set up class-imbalance weights using scheme
    freq1 = CountFrequency(np.argmax(y1, axis=1))
    freq2 = CountFrequency(np.argmax(y2, axis=1))
    freq3 = CountFrequency(np.argmax(y3, axis=1))
    class_weights = {
        'pub': class_weight_dict(freq1, mu=0.2),
        'fact': class_weight_dict(freq2, mu=0.2),
        'bias': class_weight_dict(freq3, mu=0.2)
    }

    # Split data (default set to be 70/30, ignoring 30 --> 15-15 plot)
    print("... loading training data")
    y_train = []
    y_test = []
    for y in [y1, y2, y3]:
        X_train, X_test, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=42)
        y_train.append(y_tr)
        y_test.append(y_te)

    ###########################################################

    print("... fitting the classifier model")

    # Load neural classifier model
    classifier = KerasClassifier(vec_size, N_pubs).model()
    classifier.summary()

    # Load callbacks, optimizer, loss details
    callbacks = get_callbacks(WEIGHTS_FPATH, LOG_FPATH, "val_fact_loss")

    # Use adam optimer, clip
    adam = Adam(lr=lr, clipnorm=1.0)

    # Multi-head loss. Keras adds by default
    losses = {
        "pub": 'categorical_crossentropy',
        "fact": 'categorical_crossentropy',
        "bias": 'categorical_crossentropy'
    }

    lossWeights = {
        "pub": 1.0,
        "fact": np.log(N_pubs / 3),
        "bias": np.log(N_pubs / 7)
    }

    metrics = ['accuracy']  #, top3_acc, top5_acc, top10_acc]
    classifier.compile(
        loss=losses, loss_weights=lossWeights, optimizer=adam, metrics=metrics)

    # Train the model
    estimator = classifier.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=N_epochs,
        callbacks=callbacks,
        shuffle=True,
        batch_size=batch_size,
        verbose=2,
        class_weight=class_weights,
    )

    print("... publisher: train accuracy: %.2f%% / val accuracy: %.2f%%" %
          (100 * estimator.history['pub_acc'][-1],
           100 * estimator.history['val_pub_acc'][-1]))
    print("... factual: train accuracy: %.2f%% / val accuracy: %.2f%%" %
          (100 * estimator.history['fact_acc'][-1],
           100 * estimator.history['val_fact_acc'][-1]))
    print("... bias: train accuracy: %.2f%% / val accuracy: %.2f%%" %
          (100 * estimator.history['bias_acc'][-1],
           100 * estimator.history['val_bias_acc'][-1]))

    # Breakdown of class accuracies, etc.
    #joined_dir = os.path.join(data_save_dir,'results')
    #if not os.path.isdir(joined_dir):
    #    os.makedirs(joined_dir)
    class_report(
        X_train,
        y_train,
        classifier,
        outfile=os.path.join(joined_dir, "classifier/class_report_train"))
    class_report(
        X_test,
        y_test,
        classifier,
        outfile=os.path.join(joined_dir, "classifier/class_report"))

    # Save model weights to h5
    classifier.save_weights(WEIGHTS_FPATH, overwrite=True)

    # Save model + weights to pkl for faster loading with app
    with open(WEIGHTS_FPATH + '.pkl', "wb") as fid:
        pickle.dump(classifier, fid)

    # Save results for making plots, reports, etc.
    with open(PLOT_DATA_LOC, "wb") as fid:
        pickle.dump([estimator, X_train, y_train, X_test, y_test], fid)
