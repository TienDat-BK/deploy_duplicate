from source.minHashDetection import MinHashDetection
from source.Preprocessor import *
text = ["Hello world! This is a test sentence.", 
        "Hello world! This is a test sentence.",
        "This is another sentence for testing.",
        "Completely different content here."]
sh = Shingling()
vec = sh.preprocessing(text)
print("Shingling Vectors:")
for v in vec:
    print(f"ID: {v.id}, Vector: {v.vec}")

# detector = MinHashDetection()
# clusters = detector.detect(text)

# for i, cluster in enumerate(clusters):
#     print(f"Cluster {i+1}:")
#     for vec_record in cluster:
#         print(f"  ID: {vec_record.id}, Vector: {vec_record.vec}")



