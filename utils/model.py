import pickle
import sys
import numpy as np
from gensim.models import Doc2Vec

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.regularizers import l1, l2


class doc2vec:
    '''
    Custom doc2vec class to handle gensim doc2vec model
    '''

    def __init__(self, DOC2VEC_MODEL=None, vec_size=1000, window=300,
                 min_count=10, alpha=0.025, min_alpha=0.001, cores=1):

        self.weights_path = DOC2VEC_MODEL

        self.vec_size = vec_size
        self.window = window
        self.min_count = min_count

        self.alpha = alpha
        self.min_alpha = min_alpha

        self.cores = cores

        self.load()

    def load(self):
        '''
        Load model either from weight file or initiate model
        '''
        if self.weights_path:
            self.model = Doc2Vec.load(self.weights_path)
        else:
            self.model = Doc2Vec(vector_size=self.vec_size,
                                 window=self.window,
                                 alpha=self.alpha,
                                 min_alpha=self.min_alpha,
                                 min_count=self.min_count,
                                 workers=self.cores,
                                 hs=1, sample=0, dm=0,
                                 #negative = 5,
                                 #dbow_words = 1,
                                 )

    def doc_vectors(self, tagged_docs, label_index=0, reinfer_train=False,
                    infer_steps=5, infer_alpha=None, min_words=1):
        '''
        Method to take text -> doc vector
        '''
        docvals = tagged_docs.values
        docvals = [doc for doc in docvals if len(doc.words) >= min_words]

        print("Total documents with length >= {}: {}".format(
            min_words, len(docvals))
        )

        # Force infer even if vector already in docvecs
        if reinfer_train:
            x, y1, y2, y3 = zip(*[(self.model.infer_vector(
                doc.words, steps=infer_steps),
                doc.tags[0], doc.tags[1], doc.tags[2])
                for doc in docvals]
            )
        else:
            def _get(doc):
                if label_index in doc.tags:
                    return (
                        self.model.docvecs[doc.tags[label_index]],
                        doc.tags[0], doc.tags[1], doc.tags[2]
                    )
                else:
                    return (
                        self.model.infer_vector(doc.words, steps=infer_steps),
                        doc.tags[0], doc.tags[1], doc.tags[2]
                    )
            x, y1, y2, y3 = zip(*[_get(doc) for doc in docvals])

        return np.array(x), y1, y2, y3

    def vec_size(self):
        return self.model.docvecs[0].shape[0]


class KerasClassifier:
    '''
    Classifier model that takes doc vectors as input and predicts [pub, fact, lean] labels
    '''

    def __init__(self, word_vector_dim, N_classes, N_fact=3, N_bias=7):
        self.word_vector_dim = word_vector_dim
        self.N_classes = N_classes
        self.N_fact = N_fact
        self.N_bias = N_bias

    def shared_block(self, inputs):
        x = Dense(1000, input_dim=self.word_vector_dim,
                  kernel_initializer="normal",
                  kernel_regularizer=l2(0.0))(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)
        return x

    def publisher(self, inputs):
        x = Dense(self.N_classes, kernel_initializer="normal",
                  activation="softmax", name="pub")(inputs)
        return x

    def fact(self, inputs):
        x = Dense(self.N_fact, kernel_initializer="normal",
                  activation="softmax", name="fact")(inputs)
        return x

    def bias(self, inputs):
        x = Dense(self.N_bias, kernel_initializer="normal",
                  activation="softmax", name="bias")(inputs)
        return x

    def model(self):
        _input = Input(shape=(self.word_vector_dim,))
        x = self.shared_block(_input)
        pub_out = self.publisher(x)
        fact_out = self.fact(x)
        bias_out = self.bias(x)
        model = Model(inputs=_input, outputs=[pub_out, fact_out, bias_out])
        return model


class Fake2Vec:
    '''
    Class to combine the doc2vec and classifier models into bulk model
    '''

    def __init__(self, infer_steps=20, datapath='./'):

        self.infer_steps = infer_steps
        self.datapath = datapath

        with open(self.datapath + "models/classifier/model.h5.pkl", "rb") as fid:
            self.classifier = pickle.load(fid)

        with open(self.datapath + "data/label_encoders.pkl", "rb") as fid:
            self.encoders, self.N_pub_classes = pickle.load(fid)

        DOC2VEC_MODEL = self.datapath + \
            "models/doc2vec/doc2vec-VecSize-1000_MinDoc-10_Window-300.model"

        self.doc2vec = Doc2Vec.load(DOC2VEC_MODEL)
        self.vecsize = self.doc2vec.docvecs[0].shape[0]

        self.fact_decode = self.encoders[1].inverse_transform(range(3))
        self.bias_decode = self.encoders[2].inverse_transform(range(7))

    def predict(self, X, topk=5):
        X = self.doc2vec.infer_vector(
            X, steps=self.infer_steps).reshape(-1, self.vecsize)
        pub, fact, bias = self.classifier.predict(X)
        top_pubs = np.argsort(pub[0])[-topk:][::-1]
        pubs_decode = [[self.encoders[0].inverse_transform(
            [k])[0], float("%0.2f" % (100*pub[0][k]))] for k in top_pubs]
        facts = [[decoded, float("%0.2f" % (100*fact_score))]
                 for fact_score, decoded in sorted(zip(fact[0], self.fact_decode))[::-1]]
        affiliation = [[decoded, float("%0.2f" % (100*bias_score))]
                       for bias_score, decoded in sorted(zip(bias[0], self.bias_decode))[::-1]]
        return pubs_decode, facts, affiliation

# class app_loader:
# load model in self, main a method


def main(X, infer_steps=20, path='../'):
    '''
    This is called in app/main.py any time a query is initiated. 
    and loads the model each time, wasteful. Could not figure out how 
    to resolve (apparent) threading problems b/t flask and gensims doc2vec.
    So leaving as is ... slow but not TERRIBLY slow for the user. 
    '''
    model = Fake2Vec(infer_steps=infer_steps, datapath=path)
    result = model.predict(X)
    K.clear_session()
    return result


if __name__ == '__main__':

    X = str(sys.argv[1])
    infer_steps = 20

    model = Fake2Vec(infer_steps=infer_steps)
    pubs_decode, facts, affiliation = model.predict(X)

    print("------------------------------------------------")
    print("Document similar to publisher")
    for (decoded, pub_score) in pubs_decode:
        print("...", decoded, ":", pub_score)

    print("------------------------------------------------")
    print("Factual assesment of document:")
    for (decoded, fact_score) in facts:
        print("...", decoded, ":", fact_score)

    print("------------------------------------------------")
    print("Lean/affiliation of document:")
    for (decoded, bias_score) in affiliation:
        print("...", decoded, ":", bias_score)
