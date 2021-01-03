from nltk import PorterStemmer
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

# function to process the reviews

def preprocess(df):
    print("Preprocessing ..................")
    # convert reviews to lower case.
    df["ReviewBody"] = df["ReviewBody"].apply(lambda x: x.lower())
    df["ReviewTitle"] = df["ReviewTitle"].apply(lambda x: x.lower())

    # setting the reviews as positive and negative only
    df.loc[df["ReviewStar"] < 3,"ReviewStar"] = -1;
    df.loc[df["ReviewStar"] > 3,"ReviewStar"] = 1;

    # seperating the words with special symbols
    df["ReviewBody"] = df["ReviewBody"].apply(lambda x: x.replace("."," . "))
    df["ReviewBody"] = df["ReviewBody"].apply(lambda x: x.replace("n't"," not "))

    return df;


def stemming(s,stops):
    ps = PorterStemmer();
    # performing word stemming for string s;
    ans = ""
    tokenize_sentence = sent_tokenize(s)
    for sentence in tokenize_sentence:
        words = word_tokenize(sentence)
        for word in words:
            if word not in stops:
                tag = pos_tag([word])
                if(tag[0][1] == 'NN'):
                    word_stem = ps.stem(word)
                    ans = ans + " "+word_stem + "_NN"
                elif(tag[0][1] == 'JJ' or tag[0][1] == 'JJR') or tag[0][1] == 'JJS':
                    word_stem = ps.stem(word)
                    ans = ans + " "+word_stem + "_JJ"
        ans = ans + " . "
    
    return ans


def word_stemming(df):
    print("Word stemming ........")
    stops = set(stopwords.words('english'))
    df["ReviewBody"] = df["ReviewBody"].apply(lambda x:stemming(x,stops));
    return df;


def pos(df):
    print("performing part of speech tagging .......")

    nouns = set();
    adjectives = set();
    for i in range(0,len(df)):
        text = df.iloc[i,1];
        words = word_tokenize(text)
        tags = pos_tag(words)
        for tag in tags:
            if tag[1] == 'NN':
                nouns.add(tag[0])
            elif tag[1] == 'JJ':
                adjectives.add(tag[0])
    
    
    return (nouns,adjectives);
