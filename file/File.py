import pandas as pd

try:
    from DataTransform import DataTransform
except:
    from file.DataTransform import DataTransform


class File(DataTransform):
     '''this is a class for reading csv files in pandas dataframes or exporting from a dataframe to a csv'''

     def __init__(self):
         self.filename = ''
         return

     def set_file_name(self,filename):
         self.filename = filename
         return

     def read_file(self):
         '''
         Takes the filename variable from the class and attempts to read it

         Initially it will set both the input and output df to the same value
         '''
         print('Attempting to read file.')
         try:
            self.input_df = pd.read_csv(self.filename)
            self.output_df = pd.read_csv(self.filename)
         except:
             print("file cannot be found")
         return

     def write_input_file(self, file_name):
         '''
         Exports the dataframe of the input file to csv
         '''
         try:
            self.input_df.to_csv(file_name, index = False)
         except:
             print("could not export data")
         return

     def write_output_file(self, file_name):
         '''
         Exports the dataframe of the output file to csv
         '''
         try:
            self.output_df.to_csv(file_name, index = False)
         except:
             print("could not export data")
         return
    
