import traceback
import newspaper
import pickle
import psutil
import time
import sys

from multiprocessing import Pool
from newspaper import Article, fulltext

# Set the limit for total number of articles to download
LIMIT = 1e10

def load_url(url, article = {}):
    content = Article(url)
    content.download()
    content.parse()

    article['title'] = content.title
    article['authors'] = content.authors
    article['text'] = content.text
    article['top_image'] =  content.top_image
    article['movies'] = content.movies
    article['link'] = content.url
    article['published'] = content.publish_date
    try:
        content.nlp()
        article['keywords'] = content.keywords
        article['summary'] = content.summary
    except Exception as E:
        pass

    text = article['title'].strip("\n")
    if 'keywords' in article:
        text += " ".join([x.strip("\n") for x in article['keywords']])
    if 'summary' in article:
        text += article['summary'].strip('\n')
    text += " ".join(article['text'].split("\n"))
    return text

def load_labeled_data(fileName = 'data/corpus.csv'):
    '''
        Load the CSV file with news sites into a dictionary
    '''
    newsPapers = {}
    with open(fileName) as fid:
        for line in fid.readlines()[1:]: #Skip the top line --> dict keys.
            url, url_short, media_bias_url, fact, bias = line.strip("\n").split(",")
            newsPapers[url_short] = {
                                    "link": url,
                                    "media-bias": media_bias_url,
                                    "fact": fact,
                                    "bias": bias,
                                    "N_articles": None,
                                    "articles": []}
    return newsPapers

def ReadTheNews(details, verbose = False):
    '''
        Function to ring URL, download all articles + misc. details
    '''
    try:
        company, newsPaper = details

        if verbose:
            print("Building site for ", company)

        link = newsPaper['link']
        fact = newsPaper['fact']
        bias = newsPaper['bias']
        media = newsPaper['media-bias']
        articles = newsPaper['articles']

        paper = newspaper.build(link, memoize_articles=False)
        newsPaper['N_articles'] = paper.size()

        if verbose:
            print(company, "total number of articles:", newsPaper['N_articles'])

        count = 0
        for content in paper.articles:
            if count > LIMIT:
                break
            try:
                content.download()
                content.parse()
            except Exception as E:
                print(E)
                print("Continuing...")
                continue

            article = {}
            article['title'] = content.title
            article['authors'] = content.authors
            article['text'] = content.text
            article['top_image'] =  content.top_image
            article['movies'] = content.movies
            article['link'] = content.url
            article['published'] = content.publish_date

            # Get some nlp data
            try:
                content.nlp()
                article['keywords'] = content.keywords
                article['summary'] = content.summary
            except Exception as E:
                print("Problem obtaining NLP data", E)

            # Append the article to newspaper entry
            articles.append(article)

            if verbose:
                print(count, "articles downloaded from", company, " using newspaper, url: ", content.url)
            count = count + 1

        # Save data to pkl file
        fileName = company + ".pkl"
        with open("data/raw/{}".format(fileName), "wb") as fid:
            pickle.dump([company, newsPaper], fid)

        print("... finished {} total articles {}".format(company,len(articles)))
        return 1

    except Exception as E:
        print(traceback.format_exc())
        return 0

if __name__ == '__main__':

    if len(sys.argv) > 1:
        newsPapers = load_labeled_data(fileName = sys.argv[1])
    else:
        newsPapers = load_labeled_data()

    NCPUS = psutil.cpu_count()
    pool = Pool(NCPUS)

    print("Processing data/corpus.csv ...")
    print("... Using {} CPUS".format(NCPUS))

    start = time.time()
    pool.map(ReadTheNews, newsPapers.items())
    print("Finished in {}".format(time.time()-start))
