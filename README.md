# aspect-based-sentiment-analysis
python implementation for aspect based sentiment analysis using support vector machine.

## Dataset used
Link - https://www.kaggle.com/shitalkat/amazonearphonesreviews
Amazon earphone dataset

### place the dataset files inside the dataset folder.

## Process
### Loading of Dataset.
pandas is used for loading of dataset.

### Preprocessing.
lower casing the text and body.
deviding the reviews into the classes of possitive(1) and negatve(-1) class.
stemming of words in text, using natural language toolkit.

### Feature extraction
part of speach tagging is done using nltk in python.
Frequent Nouns are used as aspects and Frequent Adjectives are used as values.
Other data mining algorithms can also be used here such as Apriori Algorithm.
##### Feature pruning is performed - removal of extra aspects and values.
##### aspects and values are made cleared from other text.
Reviews with no aspects or values are ignored.

### Features creation
"aspect_value" pairs are used for feature of the review.
Values are assigned to the aspects by applying the Breadth-first search i.e aspects are assigned to the closest value.

### Support Vector Machine is imlemented using sklearn
Other algorithms can also be used like - naive bayes, Logistic Regression.

## Result
### training result:-
Accuracy: 0.8675252989880404
Precision: 0.8651372957942302
Recall: 0.9826292933280695
### Test result:-
Accuracy: 0.8357510528778662
Precision: 0.8428796636889122
Recall: 0.9685990338164251

#### Aspect based sentiment analysis performance is bounded by natural language processing i.e open to research.
