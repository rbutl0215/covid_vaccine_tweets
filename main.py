import pandas
from file.File import File
import matplotlib

def main():
    file = File()
    file.train_test_model()
    file.set_file_name('vaccination_tweets.csv')
    file.read_file()
    file.clean_text('text','token_tweets')
    file.classify_sentiment('token_tweets','sentiment')
    file.write_output_file('tokenized_tweets.csv')
    

if __name__ == '__main__':
    main()
        