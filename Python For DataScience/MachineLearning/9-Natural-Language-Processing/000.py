import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# nltk.download_shell()
messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
# print(len(messages))
# print(messages[50])
# for mess_no, message in enumerate(messages[:10]):
#     print(mess_no, message)
#     print('\n')

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])
# print(messages.describe())
# print(messages.groupby('label').describe())
messages['length'] = messages['message'].apply(len)
# print(messages.head())

# messages['length'].hist(bins=50)
# print(messages['length'].describe())
# print(messages[messages['length'] == 910]['message'].iloc[0])
# messages.hist(column='length', by='label', bins=60, figsize=(12, 4))

############################### PART 2 ###############################
import string

mess = 'Sample message! Notice: it has punctuation.'
nopunc = [c for c in mess if c not in string.punctuation]
# print(nopunc)

from nltk.corpus import stopwords

nopunc = ''.join(nopunc)
# print(nopunc)

clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# print(clean_mess)

def text_process(mess):
    '''
    1. remove punc
    2. remove stop words
    3. return list of clean text words
    '''
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# print(messages['message'].head(5).apply(text_process))

from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
# print(len(bow_transformer.vocabulary_))

mess4 = messages['message'][3]
# print(mess4)
bow4 = bow_transformer.transform([mess4])
# print(bow4)
# print(bow4.shape)
# bow_transformer.get_feature_names()[4068]
# bow_transformer.get_feature_names()[9554]


############################### PART 3 ###############################
messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)
tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]
messages_tfidf = tfidf_transformer.transform(messages_bow)

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])
spam_detect_model.predict(tfidf4)[0]
print(messages['label'][3])
all_pred = spam_detect_model.predict(messages_tfidf)
print(all_pred)

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)
from sklearn.pipeline import  Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', tfidf_transformer),
    ('classifier', MultinomialNB())
])
pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(label_test, predictions))
print(confusion_matrix(label_test, predictions))

plt.show()
