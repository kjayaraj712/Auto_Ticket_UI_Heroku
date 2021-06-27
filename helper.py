import string
import re
from nltk.corpus import stopwords
import nltk
from textblob import Word


### Class to perform the pre-processing of the text
class Assignor():
    '''
    This class hosts all the functions required for pre-processing of the text before it can be supplied for
    training or inference/detection.
    '''
    ft_lid_model = None
    def __init__(self):
      None
        # Constructor function
    
    def __cleanup_text(self, text, RemovePatterns=[]):
      '''
      This function accepts text and a list of Reg-Ex patterns. It applies the patterns
      on the input text and replaces them with a space
      '''
      for pattern in RemovePatterns:
        pattern = re.compile(pattern)
        text = pattern.sub(' ', text)
      return ' '.join(text.split())
    
    
    def load_language_model(self, modelpath):
      '''
      This function loads the language detection model. The model path is accepted as a parameter.
      '''
      if self.ft_lid_model is None:
        import fasttext as ft
        self.ft_lid_model = ft.load_model(modelpath)
        print("Language Model loaded successfully")


    def preprocess_input(self,
                         X, 
                         RemoveNewLine=True, 
                         RemoveUTFSpecialChars = True, 
                         RemoveEmail = True, 
                         RemoveCommHeaders = True, 
                         RemoveWordsWithNumbers = True, 
                         RemoveSalutations = True, 
                         RemoveURLs = True,
                         RemoveSchedulerDateTime = True,
                         RemoveSpecifictoRow = False,
                         SpecificPatternArray = None
                        ):
      '''
      Preprocessing a numpy array of texts. This function does cleanup of un-wanted text in the
      input numpy array. If there is a specific string/pattern to be removed from each of the rows, 
      a numpy array of the same dimention can be passed.
      '''

      if RemoveSpecifictoRow:
        if SpecificPatternArray is None:
          print ('When RemoveSpecifictoRow is True, an array of tokens SpecificPatternArray must be provided. Exiting.')
          return
        if len(X) != len(SpecificPatternArray):
          print ('Length of input array and SpecificPatternArray must match')
          return

      RemovalPatterns = []

      if RemoveNewLine:
         RemovalPatterns.append(r'\r\n')

      if RemoveWordsWithNumbers:
        RemovalPatterns.append(r'\w*\d\w*')


      if RemoveUTFSpecialChars:  
        RemovalPatterns.append(r'[\x00-\x2F]')           # special characters and punctuation
        RemovalPatterns.append(r'[\x7B-\xFF]')           # special characters

      if RemoveCommHeaders:
        RemovalPatterns.append(r'received from:')                  # email headers received from: 
        RemovalPatterns.append(r'from:[\s*\w*:,\r\n]*to:\s*\w*\s*\w*') # email headers from:...... to:
      
      if RemoveEmail:
        RemovalPatterns.append(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')  # email ids
        RemovalPatterns.append(r'(@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')                  # email id domain some ids in the text are missing the id
            
      if RemoveSalutations:
        RemovalPatterns.append(r'dear [\w,]*')                  # saluations, with or without the comma
        RemovalPatterns.append(r'hello [\w,]*')
        RemovalPatterns.append(r'hi [\w,]*')

      if RemoveSchedulerDateTime:
        RemovalPatterns.append(r'at: \d\d/\d\d/\d\d\d\d \d\d:\d\d:\d\d"*')     # automated job timestamps

      if RemoveURLs:
        RemovalPatterns.append(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")

      for i in range(X.shape[0]):
      
        RowSpecificRemovalPatterns = RemovalPatterns[:]
      
        if RemoveSpecifictoRow:  
          RowSpecificRemovalPatterns.append(SpecificPatternArray[i])

        X[i] = self.__cleanup_text(X[i], RemovePatterns=RowSpecificRemovalPatterns)      


       
      return X

    def discard_languages(self,
                          X, 
                          y,
                          X2=None,
                          KeepLanguages=['en'],
                          LangConfThreshold=0.8 
                          ):
      '''
      This function accepts numpy arrays X and y for text features and labels. It determines the text language
      for the features and discards the observations that are not in the list of languages in the provided list of languages
      This can take two related text feature sets and a label set. The rows to be discared will be removed from all the arrays to
      keep them consistent.
      '''
      if (len(KeepLanguages)==0):
        return X,y

      if (self.ft_lid_model is None):
        print("Please load language model using function load_language_model")
        return

      tobediscarded_idx = []
  
      for i in range(X.shape[0]):
        lang_pred = self.ft_lid_model.predict(X[i])
        lang = lang_pred[0][0].replace('__label__','')
        langconf = lang_pred[1][0]
    
        if lang not in KeepLanguages:
          tobediscarded_idx.append(i)
        else:
          if langconf < LangConfThreshold:
             tobediscarded_idx.append(i)

      X = np.delete(X, tobediscarded_idx, axis=0)       
      y = np.delete(y, tobediscarded_idx, axis=0)       
      if X2 is not None:
        X2 = np.delete(X2, tobediscarded_idx, axis=0)       

      print(f"No of discarded texts:\t{len(tobediscarded_idx)}")
      return X, y, X2

