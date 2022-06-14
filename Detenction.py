import numpy as np
import pandas as pd
import seaborn as sns
from keras.wrappers.scikit_learn import KerasClassifier
from keras_preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt, pyplot
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, precision_score, \
    recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.corpus import stopwords
import gensim
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tensorflow.keras.preprocessing.text import Tokenizer
from keras import models, Sequential, Input, Model
from keras import layers
from keras import callbacks
from sklearn.linear_model import LogisticRegression


# Reading the csv file and storing it in a dataframe called datafake.
datafake = pd.read_csv("Fake.csv")
# Reading the csv file and storing it in a dataframe called datatrue.
datatrue = pd.read_csv("True.csv")

#Aggiunta di una nuova colonna Target
#Per il dataset delle fake news la nuova colonna viene riempita con 0
#Per il dataset delle fake news la nuova colonna viene riempita con 1

datafake['Target'] = 0
print(datafake.info())
print(datafake.value_counts('subject'))
print(datafake.isnull().sum())

datatrue['Target'] = 1
print(datatrue.info())
print(datatrue.value_counts('subject'))
print(datatrue.isnull().sum())

print(datafake['text'].sample(15))
print(datatrue['text'].sample(15))


# Concatenating the two dataframes datafake and datatrue into one dataframe called df.
df = pd.concat([datafake, datatrue], ignore_index= True)
ax = sns.countplot(x="Target", data=df, palette="Set3")
plt.show()

# Plotting the subject column of the dataframe df.
print(df.value_counts('subject'))
ax = sns.countplot(x="subject", data=df, palette="Set3")
plt.xticks(rotation=15)
plt.show()

sns.countplot(x="subject", data=df, hue = 'Target', palette="Set3")
plt.xticks(rotation=15)
plt.show()

print (df.head)

# Creating a set of stopwords in the English language.
stop_words = set(stopwords.words('english'))

# Creating a list of the first 100 stopwords.
list_of = list(stop_words)[:100]
print(list_of)

# Removing the stopwords from the text column of the dataframe df.
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
print (df.head)



def normalise_text(text):
    """
    It takes a string as input, and returns a string with all the URLs, whitespaces, and punctuation
    removed
    
    :param text: The text that needs to be cleaned
    """
    
    #re  to normalise
    # Converting all the characters in the text to lowercase.
    text = text.str.lower()
    #text = text.str.replace(r"donald",'oBAMAAAA') # test per vedere se va
    # Replacing all the URLs in the text with the word "URL".
    text = text.str.replace(r'http\S+','URL')
    text = text.str.replace(r'@','')
    text = text.str.replace(r"!", ' ')
    text = text.str.replace(r"?", ' ')
    text = text.str.replace(r"'", ' ')
    text = text.str.replace(r"(", ' ')
    text = text.str.replace(r")", ' ')
    text = text.str.replace(r"[", ' ')
    text = text.str.replace(r"]", ' ')
    # Replacing all the whitespaces with a single whitespace.
    text = text.str.replace("\s{2,}"," ")

    return text


# Concatenating the title and the text columns of the dataframe df.
df['text'] = df['title'] + " "+ df['text']
# Removing all the URLs, whitespaces, and punctuation from the text column of the dataframe df.
df['text'] = normalise_text(df['text'])
del df['title']
del df['subject']
del df['date']

print(df.sample(15))


# Creating a list of lists, where each list is a list of words in a document.
X = [d.split() for d in df['text'].tolist()]
y = df['Target'].values
print(type(X))
print ('Wait...')

DIM = 100
# Creating a word2vec model with the sentences in X, with a vector size of DIM, a window of 10, and a
# minimum count of 1.
w2v_model = gensim.models.Word2Vec(sentences=X,vector_size=DIM, window=10, min_count=1)
print (w2v_model)


print (len(w2v_model.wv.key_to_index))
print(w2v_model.wv.most_similar('twitter'))
print(w2v_model.wv.most_similar('war'))
print(w2v_model.wv.most_similar('trump'))
#print (w2v_model.wv.similarity('twitter','instagram'))
#print(w2v_model.most_similar(positive = ['trump','president'], negative =['man']))

# Da tenere commentato!!
"""words_test = ["conflict", "invasion", "trump", "war-era", "president"]

X = w2v_model.wv[words_test]
    
pca = PCA(n_components=2)
result = pca.fit_transform(X)

pyplot.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words_test):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()"""


# Creating a dictionary of words and their corresponding indices.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X= tokenizer.texts_to_sequences(X)
#print(tokenizer.word_index)
plt.hist([len (x) for x in X], bins=700)
plt.show()

# Padding the sequences with zeros to make them all the same length.
X= pad_sequences(X)

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y)


#LOGISTIC REGRESSION

from sklearn.metrics import mean_absolute_error
logreg = LogisticRegression()
# Create an instance of Logistic Regression Classifier and fit the data.
logreg.fit(X_train, y_train)
# Make predictions using the testing set
y_pred = logreg.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)
m=mean_absolute_error(y_test,y_pred)
print("errore assoluto medio: \n", m)

print("Confusion Matrix logistic regression: \n",
      confusion_matrix(y_test, y_pred))

print("Accuracy logistic regression : \n", accuracy_score(y_test, y_pred) * 100)

print("Report logistic regression : \n",classification_report(y_test, y_pred))


#DECISON TREE
clf_gini = DecisionTreeClassifier(criterion="gini", max_depth=15,min_samples_leaf=17)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
print("Predicted values:")
print(y_pred)

clf_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=15, min_samples_leaf=17)
clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)
print("Predicted values with entropy : ")
print(y_pred)


print("Confusion Matrix: \n",
      confusion_matrix(y_test, y_pred))

print("Accuracy : \n", accuracy_score(y_test, y_pred) * 100)

print("Report : \n",classification_report(y_test, y_pred))

plot_tree(clf_gini, filled=True)
plt.show()
plot_tree(clf_entropy, filled=True)
plt.show()

#RETE NEURALE
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))  # input layer
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # output layer

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

# EarlyStopping per evitare overfitting
# patience= Numero di epoche senza alcun miglioramento dopo le quali l'allenamento verrà interrotto
# min_delta =la variazione minima nella quantità monitorata per qualificarsi come miglioramento, ovvero una variazione assoluta inferiore a min_delta, non verrà conteggiata come miglioramento.
call = callbacks.EarlyStopping(patience=5, min_delta=0.0001, restore_best_weights=True)
# fare fit del modello
history = model.fit(X_train, y_train, validation_split=0.2, batch_size=60, epochs=80, callbacks=[call])

test_loss, test_ac = model.evaluate(X_test, y_test)
model.summary()
print(test_ac)

y_pred = model.predict(X_test)
y_pred = np.array(y_pred >= 0.5, dtype=np.int64)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#RICERCA IMPERPARAMETRI MIGLIORI, TEMPO DI ESECUZIONE CIRCA 3 ORE 

"""def create_model():
	# create model
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))  # input layer
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # output layer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
    return model

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20,28, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
"""


