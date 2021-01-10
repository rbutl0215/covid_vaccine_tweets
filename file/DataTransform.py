import pandas as pd
import data_describe as dd
import nltk

class DataTransform:
    
    def __init__(self):
        self.input_df = pd.DataFrame()
        self.output_df = pd.DataFrame()
        return

    def clean_text(self, column_name, new_column_name):
        self.tokenize(column_name,new_column_name)
        self.df_list_to_lower(new_column_name)
        self.df_remove_punctuation(new_column_name)
        self.df_remove_stop_words(new_column_name)
        self.df_remove_redacted(new_column_name)
        return

    #this function tokenizes a selected column from a dataframe

    def tokenize(self,column_name,new_column_name):
        print('Tokenizing...')
        self.output_df[new_column_name] = self.input_df.apply(lambda row: nltk.word_tokenize(row[column_name]), axis = 1)
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



            