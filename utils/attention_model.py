import numpy as np
import os, sys, pickle
from nltk import tokenize
#from fake2vec.utils.train_doc2vec import load_data

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model

from keras.layers import Embedding
from keras.layers import Dense, Input, Dense, Dropout, Activation, BatchNormalization
from keras.layers import Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras.optimizers import SGD, Adam
from keras.regularizers import l1, l2

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

'''
Hierarchical attention model for documents 
Some ode adapted https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py

Other useful examples - 
https://github.com/olgaliak/sequence_intent/blob/master/kaggle/kaggle_malware_han_cntk_exp.ipynb
https://github.com/minqi/hnatt/blob/master/hnatt.py
https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/

'''


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

'''
class AttentionLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttentionLayer,
              self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


'''


class AttentionLayer(Layer):
    def __init__(self, context_dim, regularizer = None):
        self.context_dim = context_dim
        self.regularizer = regularizer
        self.init = initializers.get('normal')
        self.supports_masking = True
        super(AttentionLayer, self).__init__()

    # def build(self, input_shape):
    #     assert len(input_shape) == 3
    #     self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
    #     self.b = K.variable(self.init((self.attention_dim, )))
    #     self.u = K.variable(self.init((self.attention_dim, 1)))
    #     self.trainable_weights = [self.W, self.b, self.u]
    #     super(AttentionLayer, self).build(input_shape)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(
            name='W',
            shape=(input_shape[-1], self.context_dim),
            initializer='normal',
            trainable=True,
            regularizer=self.regularizer)
        self.b = self.add_weight(
            name='b',
            shape=(self.context_dim, ),
            initializer='normal',
            trainable=True,
            regularizer=self.regularizer)
        self.u = self.add_weight(
            name='u',
            shape=(self.context_dim, ),
            initializer='normal',
            trainable=True,
            regularizer=self.regularizer)
        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(
            K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class DocAttnModel:
    def __init__(self,
                 N_words,
                 embedding_dim=100,
                 max_sentence_length=10,
                 max_number_sentences=100,
                 N_classes=938,
                 N_fact=3,
                 N_bias=7):

        self.N_words = N_words
        self.embedding_dim = embedding_dim
        self.max_sentence_length = max_sentence_length
        self.max_number_sentences = max_number_sentences

        self.N_classes = N_classes
        self.N_fact = N_fact
        self.N_bias = N_bias

        #self.classifier = self.Classifier()

    def Classifier(self):
        _input = Input(shape=(2 * self.embedding_dim, ))
        x = self.shared_block(_input)
        pub_out = self.publisher(x)
        fact_out = self.fact(x)
        bias_out = self.bias(x)
        classifier = Model(
            inputs=_input, outputs=[pub_out, fact_out, bias_out])
        classifier.summary()
        return classifier

    def shared_block(self, inputs):
        x = Dense(
            1000, kernel_initializer="normal",
            kernel_regularizer=l2(0.0))(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)
        return x

    def publisher(self, inputs):
        x = Dense(
            self.N_classes,
            kernel_initializer="normal",
            activation="softmax",
            name="pub")(inputs)
        return x

    def fact(self, inputs):
        x = Dense(
            self.N_fact,
            kernel_initializer="normal",
            activation="softmax",
            name="fact")(inputs)
        return x

    def bias(self, inputs):
        x = Dense(
            self.N_bias,
            kernel_initializer="normal",
            activation="softmax",
            name="bias")(inputs)
        return x

    def combined_model(self, embedding_matrix, nodes=100, reg=l2(1e-8), attention=True):

        ##########################################################
        #
        # Model to predict word-attention-weighted sentence scores
        #
        ##########################################################

        word_embedding = Embedding(
            self.N_words,
            self.embedding_dim,
            weights=[embedding_matrix],
            input_length=self.max_sentence_length,
            trainable=True,
            #mask_zero=True
        )

        input_sentence = Input(
            shape=(self.max_sentence_length, ), dtype='float32')
        document_embedding = word_embedding(input_sentence)
        l_lstm = Bidirectional(
            GRU(nodes, kernel_regularizer=reg,
                return_sequences=True))(document_embedding)
        if attention:
            l_lstm = AttentionLayer(nodes, regularizer=l2_reg)(l_lstm)

        encoder = Model(input_sentence, l_lstm)
        encoder.summary()

        #########################################################
        #
        # Model to predict sentence-attention-weighted doc scores
        #
        #########################################################

        pub_input = Input(
            shape=(self.max_number_sentences, self.max_sentence_length),
            dtype='float32')
        pub_encoder = TimeDistributed(
            encoder,
            input_shape=(self.max_number_sentences,
                         self.max_sentence_length))(pub_input)

        l_lstm_sent = Bidirectional(GRU(nodes, kernel_regularizer=reg,
                                        return_sequences=True))(pub_encoder)
        if attention:
            attention_weighted_doc = AttentionLayer(
                nodes, regularizer=l2_reg)(l_lstm_sent)

        pub_out = self.publisher(attention_weighted_doc)
        fact_out = self.fact(attention_weighted_doc)
        bias_out = self.bias(attention_weighted_doc)
        outputs = [pub_out, fact_out, bias_out]

        weighted_doc_encoder = Model(
            inputs=pub_input, outputs=attention_weighted_doc)
        self.sentence_attention_model = weighted_doc_encoder

        classifier = Model(inputs=pub_input, outputs=outputs)
        classifier.summary()

        return classifier

    def activation_maps(self, text, websafe=False):
        normalized_text = normalize(text)
        encoded_text = self._encode_input(text)[0]

        # get word activations
        hidden_word_encoding_out = Model(
            inputs=self.word_attention_model.input,
            outputs=self.word_attention_model.get_layer(
                'dense_transform_w').output)
        hidden_word_encodings = hidden_word_encoding_out.predict(encoded_text)
        word_context = self.word_attention_model.get_layer(
            'word_attention').get_weights()[0]
        u_wattention = encoded_text * np.exp(
            np.squeeze(np.dot(hidden_word_encodings, word_context)))
        if websafe:
            u_wattention = u_wattention.astype(float)

        # generate word, activation pairs
        nopad_encoded_text = encoded_text[-len(normalized_text):]
        nopad_encoded_text = [
            list(filter(lambda x: x > 0, sentence))
            for sentence in nopad_encoded_text
        ]
        reconstructed_texts = [[
            self.reverse_word_index[int(i)] for i in sentence
        ] for sentence in nopad_encoded_text]
        nopad_wattention = u_wattention[-len(normalized_text):]
        nopad_wattention = nopad_wattention / np.expand_dims(
            np.sum(nopad_wattention, -1), -1)
        nopad_wattention = np.array([
            attention_seq[-len(sentence):] for attention_seq, sentence in zip(
                nopad_wattention, nopad_encoded_text)
        ])
        word_activation_maps = []
        for i, text in enumerate(reconstructed_texts):
            word_activation_maps.append(list(zip(text, nopad_wattention[i])))

        # get sentence activations
        hidden_sentence_encoding_out = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('dense_transform_s').output)
        hidden_sentence_encodings = np.squeeze(
            hidden_sentence_encoding_out.predict(
                np.expand_dims(encoded_text, 0)), 0)
        sentence_context = self.model.get_layer(
            'sentence_attention').get_weights()[0]
        u_sattention = np.exp(
            np.squeeze(
                np.dot(hidden_sentence_encodings, sentence_context), -1))
        if websafe:
            u_sattention = u_sattention.astype(float)
        nopad_sattention = u_sattention[-len(normalized_text):]

        nopad_sattention = nopad_sattention / np.expand_dims(
            np.sum(nopad_sattention, -1), -1)

        activation_map = list(zip(word_activation_maps, nopad_sattention))

        return activation_map


def load_embedding(GLOVE_DIR, word_index, MAX_WORDS, EMBEDDING_DIM):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.random.random((MAX_WORDS, EMBEDDING_DIM))
    #embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and i < MAX_WORDS:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def get_callbacks(WEIGHTS_FPATH, LOG_FPATH, monitor):
    callbacks = [
        ModelCheckpoint(
            WEIGHTS_FPATH, monitor=monitor, save_best_only=True, mode='auto'),
        EarlyStopping(monitor=monitor, patience=5),
        ReduceLROnPlateau(
            monitor=monitor, factor=0.2, patience=2, min_lr=1e-7, mode='auto'),
        CSVLogger(LOG_FPATH, separator=' ', append=True),
    ]
    return callbacks


if __name__ == '__main__':

    #id	sentiment	review
    #docu_text, publisher, publisher_token, fact_score, fact_token, bias_score, bias_token
    '''
        Looking at preprocessed text at the moment - not sentences
    '''

    MAX_NB_WORDS = 100000

    MAX_SENTS = 20
    MIN_SENT_LENGTH = 5
    MAX_SENT_LENGTH = 100

    EMBEDDING_DIM = 100

    VALIDATION_SPLIT = 0.2

    lr = 0.001
    WEIGHTS_FPATH = 'models/attention/attention.h5'
    LOG_FPATH = 'training.log'

    data_save_dir = './'

    joined_dir = os.path.join(data_save_dir, 'data')
    if not os.path.isdir(joined_dir):
        os.makedirs(joined_dir)

    attention_prepared_data = os.path.join(joined_dir,
                                           'attention_preprocessed_data.pkl')
    if os.path.isfile(attention_prepared_data):
        with open(attention_prepared_data, "rb") as fid:
            pub_label, fact_label, lean_label, word_index = pickle.load(fid)
        data = np.load(attention_prepared_data + '.npy')
        N_words = len(word_index) + 1

    else:
        with open(
                os.path.join(joined_dir, 'Doc2vecTrainingDataProcessed.pkl'),
                "rb") as fid:
            results = pickle.load(fid)

        texts = []
        articles = []
        pub_label = []
        fact_label = []
        lean_label = []
        publishers = results[1]
        for pub_url in publishers:
            publisher = publishers[pub_url]
            pub_fact = fact_to_token(publisher['fact'])
            pub_bias = bias_to_token(publisher['bias'])
            pub_content = publisher['articles']

            for article in pub_content:
                title = article['title']
                summary = article['summary']
                keywords = article['keywords']
                sentences = article['sentences']

                pub_label.append(pub_url)
                fact_label.append(pub_fact)
                lean_label.append(pub_bias)

                text_str = " ".join(title) + " ".join(summary) + " ".join(
                    keywords) + " ".join([" ".join(x) for x in sentences])
                texts.append(text_str)

                combined_sentences = [title] + [summary] + [keywords
                                                            ] + sentences
                articles.append(combined_sentences)

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)

        word_index = tokenizer.word_index
        N_words = len(word_index) + 1

        print('Total of %s unique tokens.' % len(word_index))
        print("Data tensor dimension:", len(texts), MAX_SENTS, MAX_SENT_LENGTH)

        data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH),
                        dtype='int32')

        for i, sentences in enumerate(articles):
            for j, words in enumerate(sentences):
                if j < MAX_SENTS and len(words) >= MIN_SENT_LENGTH:
                    k = 0
                    for word in words:
                        if word not in tokenizer.word_index:
                            continue
                        if k < MAX_SENT_LENGTH and tokenizer.word_index[
                                word] < MAX_NB_WORDS:
                            data[i, j, k] = tokenizer.word_index[word]
                            k = k + 1

        with open(attention_prepared_data, 'wb') as fid:
            pickle.dump([pub_label, fact_label, lean_label, word_index], fid)
        np.save(attention_prepared_data + '.npy', data)

    pub_encoder = LabelEncoder()
    pub_label = pub_encoder.fit_transform(pub_label)
    pub_label = to_categorical(np.asarray(pub_label))

    fact_encoder = LabelEncoder()
    fact_label = fact_encoder.fit_transform(fact_label)
    fact_label = to_categorical(np.asarray(fact_label))

    lean_encoder = LabelEncoder()
    lean_label = lean_encoder.fit_transform(lean_label)
    lean_label = to_categorical(np.asarray(lean_label))

    print('Shape of data tensor:', data.shape)
    print('Shape of pub-label tensor:', pub_label.shape)
    print('Shape of fact-label tensor:', fact_label.shape)
    print('Shape of lean-label tensor:', lean_label.shape)

    indices = np.arange(data.shape[0])

    np.random.seed(123)
    np.random.shuffle(indices)

    data = data[indices]
    pub_label = pub_label[indices]
    fact_label = fact_label[indices]
    lean_label = lean_label[indices]

    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = [
        pub_label[:-nb_validation_samples],
        fact_label[:-nb_validation_samples],
        lean_label[:-nb_validation_samples]
    ]

    x_val = data[-nb_validation_samples:]
    y_val = [
        pub_label[-nb_validation_samples:],
        fact_label[-nb_validation_samples:],
        lean_label[-nb_validation_samples:]
    ]
    '''
        Load model  
    '''

    combined_Model = DocAttnModel(
        MAX_NB_WORDS,
        EMBEDDING_DIM,
        MAX_SENT_LENGTH,
        MAX_SENTS,
        N_classes=938,
        N_fact=3,
        N_bias=7)
    '''
        Load Glove or GoogleNews embedding weights 
    '''

    GLOVE_DIR = 'glove'
    embedding_matrix = load_embedding(GLOVE_DIR, word_index, MAX_NB_WORDS,
                                      EMBEDDING_DIM)
    model = combined_Model.combined_model(embedding_matrix)

    #model.summary()
    plot_model(model, to_file='model_arch.png', show_shapes=True)

    # Load callbacks, optimizer, loss details
    callbacks = get_callbacks(WEIGHTS_FPATH, LOG_FPATH, "model_1_loss")

    # Multi-head loss. Keras adds by default
    # losses = {
    #     "pub": 'categorical_crossentropy',
    #     "fact": 'categorical_crossentropy',
    #     "bias": 'categorical_crossentropy'
    # }

    # lossWeights = {
    #     "pub": 1.0,
    #     "fact": np.log(938. / 3),
    #     "bias": np.log(938. / 7)
    # }

    adam = Adam(lr=1e-3, clipnorm=1.0)
    sgd = SGD(lr=1e-2, nesterov=True)

    model.compile(
        loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

    print("model fitting - Hierachical attention network")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=200,
        batch_size=128,
        shuffle=True,
        callbacks=callbacks,
        verbose=1)
