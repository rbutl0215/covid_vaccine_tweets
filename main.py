import pandas
from file.File import File
import matplotlib

def main():
    file = File()
    file.set_file_name('vaccination_tweets.csv')
    file.read_file()
    file.clean_text('text','token_tweets')
    file.clean_text('user_location','token_loc')
    file.write_output_file('tokenized_tweets.csv')

if __name__ == '__main__':
    main()
        