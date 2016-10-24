"""
NLP Monsoon 2016 Project
Wikipedia Named Entity Linking
Authors: Sagar, Shantanu, Aditya
"""

import urllib
import xmltodict
from nltk.tag import StanfordNERTagger, StanfordPOSTagger
from nltk.corpus import brown
from nltk.probability import *
import sys
import os
import re
from operator import itemgetter
os.environ['no_proxy'] = '127.0.0.1,localhost'

"""
To Do
--Apostrophes (eg Delhi's) are not identified--done
--NP chunking--done
--CrossWikis dictionary for NP chunking

API Reference

http://127.0.0.1:5000/word2vec/similarity?w1=washington&w2=america
http://127.0.0.1:5000/word2vec/n_similarity?ws1=Sushi&ws1=Shop&ws2=Japanese&ws2=Restaurant
http://127.0.0.1:5000/word2vec/similarity?w1=Sushi&w2=Japanese
http://127.0.0.1:5000/word2vec/most_similar?positive=indian&positive=food[&negative=][&topn=]
http://127.0.0.1:5000/word2vec/model?word=restaurant
http://127.0.0.1:5000/word2vec/model_word_set

*use lowercase words*


Test Sentences:

Good--> Gandhi is a 1982 epic biographical film which dramatises the life of Mohandas Karamchand Gandhi, the leader of India's non-violent, non-cooperative independence movement 

Good (bad if NNs are taken)--> Gandhi is a 1982 epic biographical film which dramatises the life of Mohandas Karamchand Gandhi, the leader of India's non-violent, non-cooperative independence movement against the United Kingdom 's rule of the country during the 20th century

Good--> Paris is the capital of France

Good--> Washington was assassinated in 1799

Bhagat Singh was an Indian revolutionary socialist who was influential in the Indian independence movement.

Good--> At ten at night George Washington spoke, requesting to be "decently buried" and to "not let my body be put into the Vault in less than three days after I am dead

Close call--> At ten at night Washington spoke, requesting to be "decently buried" and to "not let my body be put into the Vault in less than three days after I am dead

Great--> Washington had a hostile attitude towards Kremlin 's war on Afghanistan

??--> Narendra
Bad--> Modi

Total Disaster--> Angry Birds is a video game

Blunder(close call)--> Delhi woke up to a heavy smog

Great (Tsunami was in 2004)--> 2005 Tsunami caused massive damage in the Indian Ocean

Total Disaster (works fine now :)--> Inferno caused massive uproar among British intellectuals
"""
__DEBUG = 0
w2vurl = "http://127.0.0.1:5000/word2vec/similarity?w1="
stanford_dir = '/home/sagar/Programs/Stanford/'

""" NER Tagger Init"""
ner_dir = stanford_dir + 'stanford-ner-2015-12-09/'
jarfile = ner_dir + 'stanford-ner.jar'
modelfile = ner_dir + 'classifiers/english.all.3class.distsim.crf.ser.gz'

sys.stderr.write("Loading NER Tagger")
nt = StanfordNERTagger(model_filename=modelfile, path_to_jar=jarfile)
sys.stderr.write("...Done\n")
# Test
# print nt.tag("Gandhi is a 1982 biographical film based on the life of
# Mohandas Karamchand Gandhi".split())

# """ POS Tagger Init"""
pos_dir = stanford_dir + 'stanford-postagger-full-2015-12-09/'
modelfile = pos_dir + 'models/english-bidirectional-distsim.tagger'
jarfile = pos_dir + 'stanford-postagger.jar'

sys.stderr.write("Loading POS Tagger")
pt = StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)
sys.stderr.write("...Done\n")
# Test
# print pt.tag("Washington died in 1900".split())


# POS Tag sentences and return a dictionary of lists
def tagPOS(string):
    entity_dict = {'NN': [], 'JJ': [], 'CD': [], 'NNP': []}
    try:
        string += ' .'
        for i in [",", ".", "\"", ";", "?"]:
            string = string.replace(i, ' ' + i)
        entities = pt.tag(string.split())
        e_class = entities[0][1]
        entity = []
        if __DEBUG:
            print entities
        for i in entities:
            if i[1] != e_class:
                if e_class[:3] in entity_dict.keys():
                    entity_dict[e_class[:2]].append(entity)
                elif e_class[:2] in entity_dict.keys():
                    entity_dict[e_class[:2]].append(entity)
                entity = []
                e_class = i[1]
            if i[1] == e_class:
                entity.append(i[0])
        return entity_dict
    except:
        return entity_dict


# NER Tag sentences and return a dictionary of lists
def tagNER(string):
    string += ' .'
    for i in [",", ".", "\"", ";", "?"]:
        string = string.replace(i, ' ' + i)
    entities = nt.tag(string.split())
    entity = []
    e_class = entities[0][1]  # entity class
    entity_dict = {'PERSON': [], 'ORGANIZATION': [], 'LOCATION': [], 'O': []}
    for i in entities:
        if i[1] != e_class:
            if e_class != 'O':
                entity_dict[e_class].append(entity)
            entity = []
            e_class = i[1]
        if i[1] == e_class:
            entity.append(i[0])
    if __DEBUG:
        print entity_dict
    return entity_dict


# Get cosine similarity between words. Make sure word2vec API server is online
def get_similarity(w1, w2, url=w2vurl):
    query = url + w1.lower() + '&w2=' + w2.lower()
    if __DEBUG:
        print query
    try:
        score = urllib.urlopen(query).read()
        return float(score)
    except:
        return 0.0


# def class_of_word(word):
#     results = xmltodict.parse(urllib.urlopen(
#         "http://lookup.dbpedia.org/api/search.asmx/PrefixSearch?QueryClass=&MaxHits=5&QueryString=" + word).read())
#     for i in range(len(results['ArrayOfResult']['Result'])):
#         if results['ArrayOfResult']['Result'][i]['Classes'] is not None:
#             for j in range(len(results['ArrayOfResult']['Result'][i]['Classes']['Class'])):
# print
# results['ArrayOfResult']['Result'][i]['Classes']['Class'][j]['Label']


# def frequency_of_words(check):
#     words = FreqDist()

#     for sentence in brown.sents():
#         for word in sentence:
#             words[word.lower()] += 1

#     print words[check]
#     print words.freq(check)


def get_candidates(named_entities, pos):
    candidates = []
    mentions = []
    context = []
    for i in named_entities.values():
        for j in i:
            mentions.append('_'.join(j))
    mentions = list(set(mentions))
    for i in pos.values():
        for j in i:
            context.append('_'.join(j))
    context = list(set(context))
    if len(mentions) == 0:
        print "Could not identify any mention. Using nouns phrases as named entities"
    for i in pos['NNP']:
            mentions.append('_'.join(i))
    if len(mentions) == 0:
        print "Still could not identify any mention. Using normal nouns as named entities"
        for i in pos['NN']:
            mentions.append('_'.join(i))
    mentions = list(set(mentions))
    print "Mentions:", mentions
    print "Context:", context
    for mention in mentions:
        candidate = xmltodict.parse(urllib.urlopen(
            "http://lookup.dbpedia.org/api/search.asmx/PrefixSearch?QueryClass=&MaxHits=5&QueryString=" + mention).read())['ArrayOfResult']['Result']
        candidates.append(candidate)
    print "Candidates"
    flat_candidates = []
    for key, val in enumerate(candidates):
        print '->', mentions[key]
        try:
            print('--> ' + val['Label'])
            flat_candidates.append(val)
        except:
            try:
                for j in val:
                    print '--> ', j['Label']
                flat_candidates.append(val)
            except:
                pass
    return mentions, flat_candidates, context


def get_score(sentence_context, summary_context):
    # Fancy method to flatten lists
    # inp_ctx = list(set(sum(pos_tags.values(), [])))
    inp_ctx = sentence_context
    sum_ctx = summary_context
    score = 0.0
    for i in inp_ctx:
        for j in sum_ctx:
            score += get_similarity(i, j)
    try:
        score = score / (len(inp_ctx) * len(sum_ctx) * 1.0)
        return score
    except:
        return 0


def score_candidates(mentions, candidates, sentence_context):
    score = []
    if len(sentence_context) <= 1:
        print "Not enough context. Scoring based on References"
    for key, val in enumerate(candidates):
        # print '->', mentions[key]
        _score = 0
        try:
            print "Scoring:", mentions[key], 'X', val['Label'],
            if len(sentence_context) <= 1:
                _score = float(val['Refcount'])
                score.append([_score, val['Label'], val['URI']])
                print _score
                continue
            summary = val['Description']
            s_tags = tagPOS(summary)
            summary_context = []
            for i in s_tags.values():
                for j in i:
                    summary_context.append('_'.join(j))
            _score = get_score(sentence_context, summary_context)
            print _score
            score.append([_score, val['Label'], val['URI']])
        except:
            for key2, val2 in enumerate(val):
                print "Scoring:", mentions[key], 'X', val2['Label'],
                if len(sentence_context) <= 1:
                    _score = float(val2['Refcount'])
                    score.append([_score, val2['Label'], val2['URI']])
                    print _score
                    continue
                print "Scoring:", mentions[key], 'X', val2['Label'],
                summary = val2['Description']
                s_tags = tagPOS(summary)
                summary_context = []
                for i in s_tags.values():
                    for j in i:
                        summary_context.append('_'.join(j))
                _score = get_score(sentence_context, summary_context)
                print _score
                score.append([_score, val2['Label'], val2['URI']])
    score = list(sorted(score, key=itemgetter(0)))
    for i in score:
        print i


def main():
    while True:
        input = raw_input('> ')
        pos = tagPOS(input)
        ner = tagNER(input)
        print pos
        mentions, candidates, context = get_candidates(ner, pos)
        linked_entity = score_candidates(mentions, candidates, context)
        # print getSimilarity(input.split()[0], input.split()[1])


if __name__ == "__main__":
    main()
