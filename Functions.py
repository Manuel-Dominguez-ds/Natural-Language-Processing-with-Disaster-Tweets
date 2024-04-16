from nltk.corpus import stopwords
import re
import string

stopwords_eng=set(stopwords.words('english'))

def remove_stopword(text):
    #print("\n > Removing stopwords.")
    text=[x for x in text.split() if x.lower() not in stopwords_eng]
    text=' '.join(text)
    text=text.lower()
    return text

def remove_URL(text):
    #print("\n > Removing URL.")
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    #print("\n > Removing HTML.")
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    #print("\n > Removing emojis.")
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punctuation(text):
    #print("\n > Removing punctuation.")
    punc = re.compile('[%s]' % re.escape(string.punctuation))
    punc = punc.sub('', text)
    return punc
    
def preprocess(text):
    text=remove_URL(text)
    text=remove_html(text)
    text=remove_emoji(text)
    text=remove_punctuation(text)
    text=remove_stopword(text)
    return text

def sum_characters(text):
    num_chr=sum([len(z) for z in text.split()])
    return num_chr
    
def sum_words(text):
    num_words=len(text.split())
    return num_words

def count_words_appeareance(text,dictionary):
    for i in text.split():
        if i not in dictionary.keys():
            dictionary.setdefault(i,1)
        else:
            dictionary[i]+=1
