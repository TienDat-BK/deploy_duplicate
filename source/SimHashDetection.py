from HSmodule import *
from source.FaissSearch import FaissSearch
from source.Preprocessor import Shingling, TextEmbedder

class SimHashDetection:
    def __init__(self):
        self.preprocessor = TextEmbedder()
        
       
        self.hasher = SimHash()
        self.outputDim = 64

        self.searcher = LSHSearch()
        self.searcher.bandSize = 3
        self.searcher.threshold = 0.1953125
        self.searcher.setDisFunc("hamming")
        

    def detect(self, ListOfText : list ):
        ListOfVecRecord = self.preprocessor(ListOfText)  

        # cấu hình hasher cho phù hợp với kích thước vector đầu vào
        sizeOfVector = len(ListOfVecRecord[0].vec)
        self.hasher.setInOutput( sizeOfVector, self.outputDim)

        listVecHashed = self.hasher.hash(ListOfVecRecord)

        clusters = self.searcher.classify(listVecHashed)

        # clustering đang là dạng list of list of VecRecord
        
        return clusters
