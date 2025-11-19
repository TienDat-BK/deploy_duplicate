# from source.minHashDetection import MinHashDetection
# from source.Preprocessor import *
# text = ["Hello world! This is a test sentence.", 
#         "Hello world! This is a test sentence.",
#         "This is another sentence for testing.",
#         "Completely different content here."]
# sh = Shingling()
# vec = sh.preprocessing(text)
# print("Shingling Vectors:")
# for v in vec:
#     print(f"ID: {v.id}, Vector: {v.vec}")

# detector = MinHashDetection()
# clusters = detector.detect(text)

# for i, cluster in enumerate(clusters):
#     print(f"Cluster {i+1}:")
#     for vec_record in cluster:
#         print(f"  ID: {vec_record.id}, Vector: {vec_record.vec}")

from HSmodule import *

vec1 = VectorRecord(vec = [1,2,3,4,5], id = 0)
vec2 = VectorRecord(vec = [1,2,3,5,2002], id = 1)
hasher = MinHash()
hasher.setInOutput(10,100)
vec = hasher.hash([vec1, vec2])
# print(vec[0].vec)
# print(vec[1].vec)
print(LSHSearch.jarcardDistance(vec[0], vec[1]))


