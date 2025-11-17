from HSmodule import *
from source.FaissSearch import FaissSearch
from source.Preprocessor import Shingling, TextEmbedder

class BloomDetection:
    def __init__(self):
        self.preprocessor = TextEmbedder()
        self.Bloom = BloomFilter(self.preprocessor.lenOfVector, self.preprocessor.lenOfVector, 10000, 0.01)

        self.hasher = SimHash()
        self.outputDim = 64
        self.searcher = FaissSearch()
        self.searcher.setDisFunc("cosine")
        self.searcher.threshold = 0.091

    def detect(self, ListOfText : list ):
        ListOfVecRecord = self.preprocessor(ListOfText)  
        ListOfVecRecord = self.Bloom.hash(ListOfVecRecord)
        # cấu hình hasher cho phù hợp với kích thước vector đầu vào
        sizeOfVector = len(ListOfVecRecord[0].vec)
        self.hasher.setInOutput( sizeOfVector, self.outputDim)

        listVecHashed = self.hasher.hash(ListOfVecRecord)

        clusters = self.searcher.classify(listVecHashed)

        # clustering đang là dạng list of list of VecRecord
        
        return clusters
