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
Accuracy: 0.8967517401392111
<br>
Precision: 0.8931174089068826
<br>
Recall: 0.9854079809410363

### Test result:-
Accuracy: 0.8649214659685864
<br>
Precision: 0.8688853247794708
<br>
Recall: 0.9730579254602605
<br>

#### Aspect based sentiment analysis performance is bounded by natural language processing i.e open to research.
