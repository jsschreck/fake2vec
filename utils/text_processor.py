import os
import sys
import glob
import re
import nltk
import pickle
import string
import unicodedata
import contractions
import inflect
import traceback

from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk import RegexpTokenizer, word_tokenize, sent_tokenize
from bs4 import BeautifulSoup


def cleanText(text):
    text = re.sub('\W+', ' ', text.lower())
    return text


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


def tokenize_str(str_):
    # keep only alphanumeric and punctations
    str_ = re.sub(r'[^A-Za-z0-9(),.!?\'`]', ' ', str_)
    # remove multiple whitespace characters
    str_ = re.sub(r'\s{2,}', ' ', str_)
    # punctations to tokens
    str_ = re.sub(r'\(', ' ( ', str_)
    str_ = re.sub(r'\)', ' ) ', str_)
    str_ = re.sub(r',', ' , ', str_)
    str_ = re.sub(r'\.', ' . ', str_)
    str_ = re.sub(r'!', ' ! ', str_)
    str_ = re.sub(r'\?', ' ? ', str_)
    # split contractions into multiple tokens
    str_ = re.sub(r'\'s', ' \'s', str_)
    str_ = re.sub(r'\'ve', ' \'ve', str_)
    str_ = re.sub(r'n\'t', ' n\'t', str_)
    str_ = re.sub(r'\'re', ' \'re', str_)
    str_ = re.sub(r'\'d', ' \'d', str_)
    str_ = re.sub(r'\'ll', ' \'ll', str_)
    # lower case
    return str_.strip().lower().split()


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = [word.lower() for word in words]
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            try:
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            except:
                # Sometimes fails, returning long numbers.
                continue
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = [
        word for word in words if word not in stopwords.words('english')
    ]
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas


def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = [word.strip('\n') for word in words]
    words = [word.split('\n') for word in words]
    words = [item for sublist in words for item in sublist]
    words = [word for word in words if not word.isdigit()]
    return words


def ProcessWebsite(fileName):
    # Use with mp.Pool
    I = Website(fileName)
    filtered_filesaveloc = fileName.split('.pkl')[0] + '.p'
    if os.path.isfile(filtered_filesaveloc):
        I.load(filtered_filesaveloc)
        return [I.company, I.url, I.fact_score, I.bias_score, I.total_content]
    I.load_newspaper(save_location=filtered_filesaveloc)
    return [I.company, I.url, I.fact_score, I.bias_score, I.total_content]


class Website:
    def __init__(self, newsFile):
        '''
            Load preprocessed '.p' file, or load '.pkl' and preprocess then save to '.p'
        '''
        self.load(newsFile)

    def clean(self, tokens=[]):
        if not tokens:
            return None
        tokens = normalize(tokens)
        return tokens

    def read_article(self, article):
        # 'title', 'authors', 'text', 'top_image', 'movies', 'link', 'published', 'keywords', 'summary'

        # 1. Title
        title = article['title']
        if title:
            # If no title, continue but set value = None ([])
            title = self.clean(title.split(" "))
        else:
            title = []

        # 2. Summary
        summary = []
        if 'summary' in article:
            summary = article['summary']
            summary = self.clean(summary.split(" "))
            if not summary:
                summary = []

        # 3. Keywords
        keywords = []
        if 'keywords' in article:
            keywords = article['keywords']
            keywords = self.clean(keywords)
            if not keywords:
                keywords = []

        # 4. Authors
        authors = article['authors']
        if not authors:
            authors = []

        # 5. Date of publication
        date_published = article['published']
        if not date_published:
            date_published = []

        # The authors and/or date published not always published.
        # There are also erroneous "articles" that capture site information ... do not accept.
        if not (authors or date_published):
            return False

        # 6. Sentences
        sentences = article['text']
        sentences = [
            self.clean(x.split(" ")) for x in sentences.split("\n") if x != ''
        ]
        if not sentences:
            sentences = []

        # Document includes title, keywords, summary, sentences
        try:
            doc_words = title + keywords + summary + sentences
        except Exception as E:
            print("WARNING", E)
            print(type(title), type(keywords), type(summary), type(sentences))

        wordlist = list(
            set([item for sublist in doc_words for item in sublist]))
        [self.unique_words.add(word) for word in wordlist]

        return title, authors, date_published, sentences, keywords, summary, wordlist

    def load_newspaper(self, save_location=''):
        skipped = 0
        tokenizer = 0
        for article in self.articles:
            result = self.read_article(article)
            if not result:
                skipped += 1
                continue
            title, authors, date_published, sentences, keywords, summary, wordlist = result
            self.total_content[tokenizer] = {
                'title': title,
                'author': authors,
                'date': date_published,
                'sentences': sentences,
                'keywords': keywords,
                'summary': summary
            }
            tokenizer += 1
        self.total_content['vocab'] = self.unique_words

        if save_location:
            data = [
                self.company, self.url, self.fact_score, self.bias_score,
                self.total_content
            ]
            self.save(data, save_location)

    def yield_articles(self):
        if not self.total_content:
            print("Have not yet loaded / cleaned articles ... doing this now")
            self.load_newspaper(save_location=filtered_filesaveloc)
            for _item in self.total_content.items():
                yield _item

    def save(self, data, save_location=''):
        with open(save_location, "wb") as fid:
            pickle.dump(data, fid)

    def load(self, save_location):
        # Load from raw data (.pkl), or preprocessed (.p).
        if save_location.split(".")[-1] == 'pkl':
            # 'link', 'media-bias', 'fact', 'bias', 'N_articles', 'articles'
            with open(save_location, "rb") as fid:
                try:
                    company, corpus = pickle.load(fid)
                except:
                    print("(A): Failed to load {}".format(save_location))
                    return

            self.company = company.split(".pkl")[0]
            self.url = corpus['link']
            self.articles = corpus['articles']
            self.fact_score = corpus['fact']
            self.bias_score = corpus['bias']

            self.unique_words = set()
            self.total_content = {}

        else:
            with open(save_location, "rb") as fid:
                data = pickle.load(fid)
                self.company, self.url, self.fact_score, self.bias_score, self.total_content = data
                self.unique_words = self.total_content['vocab']
                self.articles = []


class WordsContainer:
    def __init__(self):

        self.articles = {}
        self.words = set()
        self.current_token = 0

    def add(self, data):
        company, url, fact_score, bias_score, total_content = data
        tokens = total_content.keys()
        if not tokens:
            return
        unique_words = total_content['vocab']
        for word in unique_words:
            self.words.add(word)
        if company not in self.articles:
            self.articles[company] = {
                'fact': fact_score,
                'bias': bias_score,
                'articles': [],
                'tokens': []
            }
        for token in tokens:
            if token == 'vocab':
                continue
            self.articles[company]['articles'].append(total_content[token])
            self.articles[company]['tokens'].append(self.current_token)
            self.current_token += 1

    def save(self, fileName):
        with open(fileName, "wb") as fid:
            pickle.dump([self.words, self.articles], fid)
        size = len(self.articles.keys())
        print("... total words: {}".format(len(self.words)))
        print("... total companies: {}".format(size))
        print("... total articles: {}".format(self.current_token))
        print("... articles per company: {}".format(self.current_token / size))

    def load(self, fileName):
        with open(fileName, "rb") as fid:
            self.words, self.articles = pickle.load(fid)
        self.current_token = len(self.articles)


if __name__ == '__main__':
    from multiprocessing import Pool
    import psutil
    import time

    NCPUS = psutil.cpu_count()
    pool = Pool(NCPUS)

    # Load data files, created by scraper.py.
    fileNames = glob.glob('data/raw/*.pkl')
    if not fileNames:
        print("You need to scrape websites first. python utils/scraper.py")
        sys.exit()

    print("Processing/loading {} news-websites ...".format(len(fileNames)))
    print("... using {} CPUS".format(NCPUS))
    start = time.time()
    results = pool.map(ProcessWebsite, fileNames)
    print("... finished in {}".format(time.time() - start))

    print("Saving all articles + labels / words into one file ... ")
    start = time.time()
    dataFrame = WordsContainer()
    for result in results:
        dataFrame.add(result)
    dataFrame.save('data/Doc2vecTrainingDataProcessed.pkl')

    print("... finished in {}".format(time.time() - start))
