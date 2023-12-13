import spacy
from spacy.tokens import DocBin
import pickle

nlp = spacy.blank("en")
training_data = [
  ("Tokyo Tower is 333m tall.", [(0, 11, "BUILDING")]),
]

# load data
trainData = pickle.load(open('./data/TrainData.pickle', mode='rb'))
testData = pickle.load(open('./data/TestData.pickle', mode='rb'))



# the DocBin will store the example documents
db = DocBin()
for text, annotations in trainData:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db.add(doc)
db.to_disk("./data/train.spacy")

# the DocBin will store the example documents
db = DocBin()
for text, annotations in testData:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db.add(doc)
db.to_disk("./data/test.spacy")