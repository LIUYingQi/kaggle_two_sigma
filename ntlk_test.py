from nltk.corpus import wordnet

a = wordnet.synset("washer/dryer in bilding")
b = wordnet.synset("washer/dryer in unit")
print a.wup_similarity(b)