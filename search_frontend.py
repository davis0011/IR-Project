from flask import Flask, request, jsonify

from inverted_index_gcp import InvertedIndex
import numpy as np
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql import *
from pyspark.sql.functions import *
import os
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import pickle
from collections import Counter
from datetime import datetime as dt


nltk.download('stopwords')


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# load the pickle files for all dictionaries used during runtime into memory
st = dt.now()
with open("meta_dict.pkl", 'rb') as f:
    meta_dict = pickle.load(f)
with open("pageviews.pkl", 'rb') as f:
    pageviews = pickle.load(f)
with open("pagerank.pkl", 'rb') as f:
    pagerank = pickle.load(f)
with open("cosine_len_dict.pkl", 'rb') as f:
    cosine_len_dict = pickle.load(f)
print("read the pickles after ", dt.now() - st)
# load the Inverted Indexes into memory
body_index = InvertedIndex.read_index("/home/shavitda/body_index", "index")
body_index.posting_locs = dict(body_index.posting_locs)
title_index = InvertedIndex.read_index("/home/shavitda/title_index", "index")
title_index.posting_locs = dict(title_index.posting_locs)
anchor_index = InvertedIndex.read_index("/home/shavitda/anchor_index", "index")
anchor_index.posting_locs = dict(anchor_index.posting_locs)
print("read the indecies after ", dt.now() - st)
# precalculated values for dictionary key misses
d_avg = 319.5242353411845
pr_avg = 0.2515575284859404
pv_avg = 15


# This code is deprecated, it was used for Word2Vec which did not make it into the final version
# print(api.load('glove-wiki-gigaword-100', return_path=True))
# model = api.load("glove-wiki-gigaword-100")


@app.route("/load")
def load():
    """Load the Inverted Indexes and pickles into the instance storage from gcloud"""

    for index in ["body_index", "title_index", "anchor_index"]:
        os.system(f"rm -r /home/shavitda/{index}/")
        os.system(f"gcloud storage cp gs://david-ir-bucket/{index} /home/shavitda/ -r")  # TODO to where
    os.system("gcloud storage cp gs://david-ir-bucket/*.pkl /home/shavitda")

    return jsonify(["done"])


RE_WORD = None
all_stopwords = None


def tokening(text):
    """

    :param text: string to tokenize according to the tokenizer and stopword removel from Assignment 3
    :return: A list of tokens, contains repetitions
    """
    global RE_WORD,all_stopwords
    if RE_WORD is None:
        RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    if all_stopwords is None:
        english_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]

        all_stopwords = english_stopwords.union(corpus_stopwords)
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return [x for x in tokens if x not in all_stopwords]


def myread(index, word):
    """
    An altered version of the read method from InvertedIndex.MultiFileReader  to support fast lookup of words

    :param index: The Inverted Index containing the word
    :param word: String to get the posting list for
    :return: List of postings in the shape [(doc_id,tf),...]
    """
    f_name, offset, n_bytes = index.posting_locs[word][0][0], index.posting_locs[word][0][1], index.df[word] * 6
    with open(index.name + "/" + f_name, 'rb') as f:
        mylist = []
        f.seek(offset)
        for i in range(int(n_bytes / 6)):
            b = (f.read(6))
            doc_id = int.from_bytes(b[0:4], 'big')
            tf = int.from_bytes(b[4:], 'big')
            mylist.append((doc_id, tf))
        return mylist


"""deprecated code to preload Word2Vec"""
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# import gensim.downloader as api
# from gensim.models.word2vec import Word2Vec
# import json
# info = api.info()
# wiki_info = api.info('glove-wiki-gigaword-100')
# # print(json.dumps(wiki_info, indent=4))
# api.load('glove-wiki-gigaword-100', return_path=True)
# model = api.load("glove-wiki-gigaword-100")
# model.most_similar("glass")
# query = "best marvel movie".split()
# for word in query:
#   print("####\n",word)
#   print(sorted(model.most_similar(word),key=lambda x:x[1], reverse=True))


###########################

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    """Load globals to avoid loading pickles and indexes into memory at query time"""
    global body_index, meta_dict, d_avgm, title_index, model
    query = tokening(query)
    if len(query) == 0:
        return jsonify(res)
    query = [q for q in query if
             q in body_index.posting_locs or q in title_index.posting_locs]
    c = Counter(query)
    main_array = list(c.keys())
    """attempts at query expansion"""
    # l = []
    # print(f"query with no stopwords is: {query}")
    # t = dt.now()
    # for q in query:
    #     l += model.similar_by_word(q, topn=5)
    # print([x[0] for x in l])
    # reverse_l = []
    # for q in l:
    #     tq = tokening(q[0])
    #     if tq == []:
    #         continue
    #     reverse_l += model.similar_by_word(tq[0])
    # prefinal = [x for x in Counter([x[0] for x in reverse_l] + query + query + query + query + query).most_common()]
    # print(f"prefinal is {prefinal}")
    # final = [prefinal[0]]
    # i = 1
    # while final[-1][1] > final[0][1] / 2:
    #     final.append(prefinal[i])
    #     i += 1
    # print(f"final is {final}")
    # print(dt.now() - t)
    # c = Counter()
    # for fin in final:
    #     if fin[0] in body_index.posting_locs or fin[0] in title_index.posting_locs:
    #         c[fin[0]] = fin[1]
    # main_array = list(c.keys())
    # expand = 2 if len(main_array) > 6 else len(main_array)
    # stemmed_query = set([stemmer.stem(word) for word in main_array])
    # from itertools import chain
    # expand_candidates = list(chain.from_iterable([model.most_similar(word) for word in main_array]))
    # expand_candidates = [x for x in expand_candidates if x[0] in body_index.posting_locs]
    # expand_stemchecked = sorted([x for x in expand_candidates if stemmer.stem(x[0]) not in stemmed_query],key=lambda x:x[1], reverse=True)
    # for ex in expand_stemchecked:
    #     expand -= 1
    #     c[(tokening(ex[0])[0])] = ex[1]/2
    #     if expand == 0:
    #         break
    # main_array = list(c.keys())
    # print("Old Q: ", query)
    # print("New Q: ", main_array)

    b = 0.75
    k = 1.5
    q_array = np.asarray([[(k + 1) * x[1] / (k + x[1])] for x in c.items()])
    n = len(main_array) + 1
    N = len(meta_dict)
    seenIDS = {}
    buckets = 10000
    candidates = [np.array([np.zeros(n, np.float32), np.zeros(n, np.float32)]) for x in range(buckets)]
    c1 = Counter()
    """
    Make a collection of matricies representing partial BM25 calculations for the relevant documents.
    The matricies will later be merged to support numpy's efficnet matrix multiplication.
    Candidate documents are decided by checking the posting list for each word in the query.
    Candidates are then added to a "bucket" in the {candidates} list according to their hashed id.
    Values in the cells of the matrix are set according to BM25, see report file for example.    
    """
    for word in main_array:
        i = main_array.index(word)
        pl = myread(body_index, word)
        for post in pl:
            if post[0] in meta_dict:
                """
                BM25 calculations
                """
                place = post[0] % buckets
                B = (1 - b) + (b * (meta_dict[post[0]][2] / d_avg))
                if post[0] not in seenIDS:
                    pos = seenIDS[post[0]] = len(candidates[place])
                    candidates[place] = np.vstack((candidates[place], np.zeros(n, np.int32)))
                    candidates[place][pos, 0] = post[0]
                    candidates[place][pos, i + 1] = (post[1] * (k - 1)) / ((B * k) + post[1]) * np.log10(
                        (N + 1) / body_index.df[word])  # term tf in doc i

                else:
                    pos = seenIDS[post[0]]
                    candidates[place][pos, i + 1] = (post[1] * (k - 1)) / ((B * k) + post[1]) * np.log10(
                        (N + 1) / body_index.df[word])
    else:
        """
        Matrix stacking and removal of paddings from submatricies
        """
        for i in range(buckets):
            candidates[i] = np.delete(candidates[i], [0, 1], 0)
        stacked = np.vstack(candidates)
        ids = stacked[:, 0]
        stacked = np.delete(stacked, 0, 1)
        res = np.matmul(stacked, q_array)
        res = res.T[0]
        res = res.astype('float32')
        m = int(res.max())
        m = m if m != 0 else 1
        res = res / m
        for x in np.vstack([ids, res]).T:
            c1[x[0]] = x[1]
    """
    Calculation of similarity of query to titles
    """
    c2 = Counter()
    for word in main_array:
        if word in title_index.posting_locs:
            res = myread(title_index, word)
            for post in res:
                c2[post[0]] += 1
    title_max = np.array(list(c2.values())).max()
    t = dt.now()
    """
    Combination of BM25 and title weights with normalization
    """
    for key in c2:
        c2[key] = c2[key] / title_max * 2
    c3 = c1 + c2
    """More deprecated code for weighting the results by their pageviews and pagerank"""
    # for key in c3:
    #     if key not in meta_dict:
    #         continue
    #     key = int(key)
    #     pr = pagerank[key] if key in pagerank else pr_avg
    #     pv = pageviews[key] if key in pageviews else pv_avg
    #     weight = (np.log10(np.log(pv))+np.log10(np.log(pr)))/2
    #     if weight < 0.1:
    #         weight = 0.1
    #     weight = weight if weight != 0 else 0.1
    #     c3[key] = c3[key]/weight
    print(len(c3), dt.now() - t)
    ret = [(int(x[0]), meta_dict[x[0]][0]) for x in c3.most_common(50)]

    # END SOLUTION
    return jsonify(ret)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    t1 = dt.now()
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    global body_index
    '''
    get candidate ids and term frequencies
    get all candidate data from meta_dict
    calc cosine sim between each doc in candidates to the query
    return top 100 sorted
    can
    '''
    global meta_dict
    candidates = {}
    c = Counter(tokening(query))
    main_array = list(c.keys())
    q_array = np.asarray(list(c.values()))
    q_size_sq = (q_array.sum()) ** 2
    n = len(main_array)
    N = len(meta_dict)
    for word in main_array:
        i = main_array.index(word)
        res = myread(title_index, word)
        for post in res:
            if post[0] not in candidates:
                candidates[post[0]] = np.zeros(n, int)
                candidates[post[0]][i] = post[1] * np.log10(N / body_index.df[word])
            else:
                candidates[post[0]][i] = post[1] * np.log10(N / body_index.df[word])

    final_countdown = Counter()
    for id in candidates:
        if id not in meta_dict:
            print(f"WHY THE FUCK IS {id} NOT IN PAGES")
            continue
        final_countdown[id] = np.dot(candidates[id], q_array) / np.sqrt((meta_dict[id][2] ** 2) * q_size_sq)
    ret = [(x[0], meta_dict[x[0]][0]) for x in final_countdown.most_common(100)]
    # END SOLUTION
    print("solution took:", dt.now() - t1)
    return jsonify(ret)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    t1 = dt.now()
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokens = set(tokening(query))
    index = InvertedIndex()
    global title_index
    counter_of_docs = Counter()
    for word in tokens:
        res = myread(title_index, word)
        for post in res:
            counter_of_docs[post[0]] += 1

    global meta_dict

    d = [(x[0], meta_dict[x[0]][0]) for x in counter_of_docs.most_common() if
         x[0] in meta_dict]  # sorted ids by match to query
    # END SOLUTION
    print("solution took:", dt.now() - t1)
    # os.system("cd ../")
    return jsonify(d)  # TODO explain noam


@app.route("/ping")
def ping():
    return jsonify(["pong"])


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    t1 = dt.now()
    res = []
    query = request.args.get('query', '')

    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokening(query)
    global anchor_index
    counter_of_docs = Counter()

    for word in tokens:
        res = myread(anchor_index, word)
        for post in res:
            counter_of_docs[post[0]] += 1

    d = [x[0] for x in counter_of_docs.most_common()]  # sorted ids by match to query
    global meta_dict
    d = [(x[0], meta_dict[x[0]][0]) for x in counter_of_docs.most_common() if x[0] in meta_dict]

    #
    # END SOLUTION
    print("solution took:", dt.now() - t1)
    return jsonify(d)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = [pagerank[wid] if wid in pagerank else 0.00001 for wid in wiki_ids]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = [pageviews[wid] if wid in pageviews else 1 for wid in wiki_ids]
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)

