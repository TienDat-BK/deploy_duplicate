import faiss
import numpy as np
from HSmodule import VectorRecord, DSU

class FaissSearch:
    def __init__(self):
        self.threshold = 0.8

        
        self.index = faiss.IndexHNSWFlat(384, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = 64 #Số lượng node được duyệt khi tìm kiếm

    def classify(self, setOfVecRecord : list[VectorRecord]) -> list[list[VectorRecord]]:
        if not setOfVecRecord:
            return []
        dim = len(setOfVecRecord[0].vec)
        self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        
        # Thêm các VectorRecord vào faiss
        n = len(setOfVecRecord)
        vecs = np.stack([v.vec for v in setOfVecRecord]).astype("float32")
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        self.index.reset()
        self.index.add(vecs)

        # Faiss search trả về độ tương đồng và index của k-nearest neighbors (Sử dụng cosine similarity)
        k = min(100, n) # k không vượt quá số vector
        similarities, indices = self.index.search(vecs, k)
        dsu = DSU(n)
        
        for i in range(n):
            for j, sim in zip(indices[i], similarities[i]):
                #Nếu vector được so sánh là chính nó hoặc không tìm thấy vector nào khác thì bỏ qua
                if i == j or j == -1:
                    continue

                # Nếu độ tương đồng vượt ngưỡng -> gộp nhóm
                if sim > self.threshold:
                    dsu.unionSet(i, j)

        groups_idx = dsu.getGroups()
        groups = [[setOfVecRecord[i] for i in group] for group in groups_idx if group]
        return groups


        
