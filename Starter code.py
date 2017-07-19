

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import spacy
from itertools import chain
from sklearn.pipeline import Pipeline

f1 = './data/training_text'
f2 = './data/training_variants'

df1 = pd.read_csv(f1, delimiter='||')
df2 = pd.read_csv(f2)


df = df1.merge(df2)

#load spacy 
nlp = spacy.load('en_core_web_md') # 'en_default' = "en_core_web_md" but stop word isn't working.
#add stop words from en_default
nlp.vocab.add_flag(lambda s: s in spacy.en.word_sets.STOP_WORDS, spacy.attrs.IS_STOP)

#nlp 
df['docs'] = [doc for doc in nlp.pipe(df['text'], batch_size=500, n_threads=4)]
print('using {0} samples'.format(len(df)))

###tokenize and remove stop words

#function to remove stop words and punctuations -- you can follow the same steps to remove other things
def filter_token(tok):
    return tok.is_stop or tok.is_punct or tok.pos_ in ["PUNCT"]\
            or tok.lower_ in ENGLISH_STOP_WORDS

#df['tokens'] contain tokenized documents			
df['tokens'] = [[tok.lower_ for tok in doc if not filter_token(tok)]
                    for doc in df['docs']]  			

## you can also use SKlearn or NLTK tokenizer/nlp functions

## look at freqcounts of training data and Y

#count of words
all_words = chain.from_iterable([words for rownum, words in df['tokens'].iteritems()])
words = pd.Series(list(all_words)).value_counts()
words.to_csv('C:/Users/Kim.Vuong/Documents/DataScienceChallenge/growinginstability/data/full_freq_tokens.csv',header=True, index=True, encoding='utf-8')

#count of labels

print(df['Class'].value_counts)

#####create X and Y training data 

#%% X and Y Labels
le = preprocessing.LabelEncoder()
le.fit(df2['Class'])
df2['Class2'] = le.transform(df2['Class'])

Y_train = df2['Class2']

#%%

X_train = df['tokens']
#X2 = df2['tokens']

docs_train, docs_test, labels_train, labels_test = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=42,stratify=Y_train)
		
def tok(x):
    return x

def prep(x):
    return x

vectorizer = TfidfVectorizer(tokenizer=tok, preprocessor=prep,
                             ngram_range=(1,1), min_df=2)


check = vectorizer.fit_transform(X_train) 
#check = check.toarray()
feature_names = vectorizer.get_feature_names()
#the curse of dimensionality or the predictors >> no. samples
print("n_samples: %d, n_features: %d" % check.shape)
    
model = Pipeline([
            ('vectorizer', vectorizer),
            #('LR',LogisticRegression(multi_class='multinomial',solver='newton-cg'))
            ])
    
model.fit(docs_train, labels_train)

labels_predict = model.predict(docs_test)
    
proba = model.predict_proba(docs_test)		