import pandas as pd
import data_describe as dd
import nltk
from nltk import FreqDist, classify, NaiveBayesClassifier

try:
    from Lemmatize import Lemmatize
except:
    from file.Lemmatize import Lemmatize

class DataTransform:
    
    def __init__(self):
        self.input_df = pd.DataFrame()
        self.output_df = pd.DataFrame()
        self.trained_model = 0
        return

    def set_df(self,new_df):
        self.input_df = new_df
        self.output_df = new_df
        return

    def clean_text(self, column_name, new_column_name):
        self.tokenize(column_name,new_column_name)
        self.df_list_to_lower(new_column_name)
        self.df_remove_punctuation(new_column_name)
        self.df_remove_stop_words(new_column_name)
        self.df_remove_redacted(new_column_name)
        self.df_lemmatize(new_column_name)
        return

    #this function tokenizes a selected column from a dataframe

    def tokenize(self,column_name,new_column_name):
        print('Tokenizing: ' + column_name )
        self.output_df[new_column_name] = self.input_df.apply(lambda row: nltk.word_tokenize(str(row[column_name])), axis = 1)
        return

    #this function lower cases as tokenized list of strings from a dataframe

    def df_list_to_lower(self,column_name):
        print('Formatting...')
        for i in range(len(self.output_df.index)):
            for j in range(len(self.output_df[column_name][i])):
                self.output_df[column_name][i][j] = self.output_df[column_name][i][j].lower()
        return

    #this functions removes punctuation from a tokenized list

    def df_remove_punctuation(self,column_name):
        for i in range(len(self.output_df.index)):
            for j in range(len(self.output_df[column_name][i])):
                if not self.output_df[column_name][i][j].isalpha():
                    self.output_df[column_name][i][j] = 'XFILTEREDX'
        return
    
    #this functions removes stop words from a tokenized list

    def df_remove_stop_words(self,column_name):
        from nltk.corpus import stopwords
        stop_words=set(stopwords.words("english"))

        for i in range(len(self.output_df.index)):
            for j in range(len(self.output_df[column_name][i])):
                if self.output_df[column_name][i][j] in stop_words:
                    self.output_df[column_name][i][j] = 'XFILTEREDX'
        return

    #this functions cleans any redacted information from previous filters in a tokenized list

    def df_remove_redacted(self,column_name):
        for z in range(20):
            for i in range(len(self.output_df.index)):
                for w in self.output_df[column_name][i]:
                    if w == 'XFILTEREDX' or w == 'https':
                        self.output_df[column_name][i].remove(w)
        return


    def df_lemmatize(self,column_name):

    # this function will lemmatize  list of strings - WIP

        print("Lemmatizing...")
        for i in range(len(self.output_df.index)):
            self.output_df[column_name][i] = Lemmatize.lemmatize(self.output_df[column_name][i])
        return

    def classify_sentiment(self,column_name, new_column_name):
        dict_samp = {}
        self.output_df[new_column_name] = 0

        for i in range(len(self.output_df.index)):
            for j in range(len(self.output_df[column_name][i])):
                dict_samp.update({self.output_df[column_name][i][j] : True})
            self.output_df[new_column_name][i] = self.trained_model.classify(dict_samp)
            dict_samp = {}
            
        return

    def train_test_model(self):
        '''
        This functions is an entirely self contained, trained Naive Bayes Model for text sentiment analysis with a 75.467% accuracy

        Importing more positive and negative classified tweets could be used to improve the model.

        The results are stored in the self.trained_model variable for the DataTransform class
        '''

        print('Preprocessing classified tweets for model.')
        from nltk.corpus import twitter_samples
        import random

        positive_tweets = twitter_samples.strings('positive_tweets.json')
        negative_tweets = twitter_samples.strings('negative_tweets.json')

        positive_df = pd.DataFrame(positive_tweets).rename(columns={0: 'text'})
        negative_df = pd.DataFrame(negative_tweets).rename(columns={0: 'text'})

        dict_samp = {}
        positive_dict = []
        positive = []
        negative=[]
        negative_dict = []

        datatransform_positive = DataTransform()
        datatransform_positive.set_df(positive_df)
        datatransform_positive.clean_text('text','token_text')

        for i in range(len(datatransform_positive.output_df.index)):
            for j in range(len(datatransform_positive.output_df['token_text'][i])):
                dict_samp.update({datatransform_positive.output_df['token_text'][i][j]: True})
            positive_dict.append(dict_samp)
            dict_samp = {}
        
        for w in positive_dict:
            positive.append((w, 'Positive'))

        datatransform_negative = DataTransform()
        datatransform_negative.set_df(negative_df)
        datatransform_negative.clean_text('text','token_text')

        for i in range(len(datatransform_negative.output_df.index)):
            for j in range(len(datatransform_negative.output_df['token_text'][i])):
                dict_samp.update({datatransform_negative.output_df['token_text'][i][j]: True})
            negative_dict.append(dict_samp)
            dict_samp = {}
        
        for w in negative_dict:
            negative.append((w, 'Negative'))

        dataset = positive+negative

        random.shuffle(dataset)

        train_data = dataset[:7000]
        test_data = dataset[7000:]

        self.trained_model = NaiveBayesClassifier.train(train_data)

        print("Accuracy is:", classify.accuracy(self.trained_model, test_data))
        return




            