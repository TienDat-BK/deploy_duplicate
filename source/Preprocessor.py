from HSmodule import *
from sentence_transformers import SentenceTransformer
import mmh3

from ftfy import fix_text
import re

# Chuẩn hóa văn bản
def normalizing(text: str):
        text = fix_text(text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.lower()
        return text

class Shingling:
  def __init__(self, k : int = 3):
    self.k = k

  def preprocessing(self, texts : list):
        # cho ra output la vector shingle
        listVecShingle = []
        mask = (1 << 40) - 1
        for text in texts:
            # chuẩn hóa văn bản
            text = normalizing(text)

            
            st = set()
            for i in range(len(text) - self.k + 1):
                shingle = text[i:i + self.k]
                st.add(mmh3.hash64(shingle)[0] & mask)
            
            listVecShingle.append(list(st))
        listVecRecord = [VectorRecord(vec = v, id = id ) for id, v in enumerate(listVecShingle)]
        return listVecRecord
  
  def __call__(self, text : list):
    return self.preprocessing(text)

class TextEmbedder:
  def __init__(self, model_name : str = 'multi-qa-MiniLM-L6-cos-v1'):
    self.model = SentenceTransformer(model_name)
    self.lenOfVector = self.model.get_sentence_embedding_dimension()

  def preprocessing(self, texts : list):
        # input list(string) - output list(VectorRecord)
        # chuẩn hóa văn bản
        texts = [normalizing(t) for t in texts]

        embeddings = self.model.encode(texts)   #O(n)    n là tổng độ dài toàn bộ văn bản
        listVecRecord = [VectorRecord(vec = emb, id = id ) for id, emb in enumerate(embeddings)]   #O(m)

        return listVecRecord
  
  def __call__(self, text : list):
    return self.preprocessing(text)
