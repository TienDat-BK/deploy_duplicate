import faiss
import numpy as np
from HSmodule import VectorRecord, DSU

class FaissSearch:
    def __init__(self):
        self.threshold = None
        self.index = None
        self.metric = None # "cosine" hoặc "hamming"
        self.dim = 384
        self.bbit = 8

    def createIndex(self, metric: str):
        if (metric == "cosine"):
            # Tạo 1 đồ thị HNSW trong không gian 384 chiều, mỗi node có tối đa 32 cạnh
            self.index = faiss.IndexHNSWFlat(self.dim, 32)
            self.index.metric_type = faiss.METRIC_INNER_PRODUCT 
            self.index.hnsw.efSearch = 64 #Số lượng node được duyệt khi tìm kiếm
            self.threshold = 0.2
        elif metric == "hamming":
            # Với b-bit minhash → binary vector, tổng số bit = dim * bbit
            n_bits = self.dim * self.bbit
            self.index = faiss.IndexBinaryFlat(n_bits)
            self.threshold = 0.4 * n_bits
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
    def cosineDistance(self, vecs: np.ndarray, k: int): 
        vecs = vecs.astype("float32")
        self.index.reset()
        self.index.add(vecs)
        k = min(k, len(vecs))
        sims, idxs = self.index.search(vecs, k)
        dists = 1 - sims
        return dists, idxs
    
    def HammingDistance(self, vecs: list[np.ndarray], k: int):
        vecs = np.stack(vecs).astype("uint8")
        self.index.reset()
        self.index.add(vecs)
        k = min(k, len(vecs))
        dists, idxs = self.index.search(vecs, k)
        return dists, idxs

    def classify(self, setOfVecRecord : list[VectorRecord]) -> list[list[VectorRecord]]: 
        if not setOfVecRecord:
            return []
        
        n = len(setOfVecRecord)
        self.createIndex(self.metric) #Khởi tạo faiss index theo metric đã gán
        self.dim = len(setOfVecRecord[0].vec)
        if self.metric is None:
            raise ValueError("Metric not set. Call createIndex(metric) before classify().")

        if self.metric == "cosine":
            vecs = np.stack([v.vec for v in setOfVecRecord])
            dists, idxs = self.cosineDistance(vecs, k=100)

        elif self.metric == "hamming":
            vecs = [v.vec for v in setOfVecRecord]
            dists, idxs = self.hammingDistance(vecs, k=100)

        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        dsu = DSU(n)
        for i in range(n):
            for j, dist in zip(idxs[i], dists[i]):
                #Nếu vector được so sánh là chính nó hoặc không tìm thấy vector nào khác thì bỏ qua
                if i == j or j == -1:
                    continue

                if dist < self.threshold:
                    dsu.unionSet(i, j)

        groups_idx = dsu.getGroups()
        groups = [[setOfVecRecord[i] for i in group] for group in groups_idx if group]
        return groups


        
