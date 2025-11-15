# import faiss
# import numpy as np
# from HSmodule import VectorRecord, DSU

# class FaissSearch:
#     def __init__(self):
#         self.threshold = 0
#         self.index = None
#         self.metric = None # "cosine" hoặc "hamming"
#         self.dim = 384
#         self.bbit = 8

#     def setDisFunc(self, metric: str):
#         self.metric = metric
#         if (metric == "cosine"):
#             # Tạo 1 đồ thị HNSW trong không gian 384 chiều, mỗi node có tối đa 32 cạnh
#             self.index = faiss.IndexHNSWFlat(self.dim, 32)
#             self.index.metric_type = faiss.METRIC_INNER_PRODUCT 
#             self.index.hnsw.efSearch = 64 #Số lượng node được duyệt khi tìm kiếm
#         elif metric == "hamming":
#             # Với b-bit minhash → binary vector, tổng số bit = dim * bbit
#             n_bits = self.dim * self.bbit
#             self.index = faiss.IndexBinaryFlat(n_bits)
#         else:
#             raise ValueError(f"Unsupported metric: {metric}")
        
#     def cosineSimilarity(self, vecs: np.ndarray, k: int): 
#         vecs = vecs.astype("float32")
#         faiss.normalize_L2(vecs) 
#         self.index.reset()
#         self.index.add(vecs)
#         k = min(k, len(vecs))
#         sims, idxs = self.index.search(vecs, k)
#         sims = (sims - sims.min()) / (sims.max() - sims.min())
#         return sims, idxs
    
#     def hammingDistance(self, vecs: list[np.ndarray], k: int):
#         # Chỉ lấy 8 bit cuối mỗi số, chuyển trực tiếp thành uint8
#         bin_array = np.stack([np.array([x % 256 for x in v], dtype=np.uint8) for v in vecs])

#         if self.index is None:
#             self.index = faiss.IndexBinaryFlat(bin_array.shape[1]*8)

#         self.index.reset()
#         self.index.add(bin_array)
#         k = min(k, len(vecs))
#         dists, idxs = self.index.search(bin_array, k)
#         return dists, idxs

#     def classify(self, setOfVecRecord : list[VectorRecord]) -> list[list[VectorRecord]]: 
#         if not setOfVecRecord:
#             return []
        
#         n = len(setOfVecRecord)
#         self.dim = len(setOfVecRecord[0].vec)

#         if self.metric is None:
#             raise ValueError("Metric not set. Call setDisFunc(metric) before classify().")

#         if self.metric == "cosine":
#             vecs = np.stack([v.vec for v in setOfVecRecord])
#             sims, idxs = self.cosineSimilarity(vecs, k=100)

#         elif self.metric == "hamming":
#             vecs = [v.vec for v in setOfVecRecord]
#             dists, idxs = self.hammingDistance(vecs, k=100)

#         else:
#             raise ValueError(f"Unsupported metric: {self.metric}")

#         sims = 1 - sims  # Chuyển similarity thành distance

#         dsu = DSU(n)
#         for i in range(n):
#             if self.metric == "cosine":
#                 for j, sim in zip(idxs[i], sims[i]):
#                     if i == j or j == -1:
#                         continue
#                     if sim <= self.threshold:
#                         dsu.unionSet(i, j)
#             elif self.metric == "hamming":
#                 for j, dist in zip(idxs[i], dists[i]):
#                     if i == j or j == -1:
#                         continue
#                     if dist <= self.threshold:
#                         dsu.unionSet(i, j)

#         groups_idx = dsu.getGroups()
#         groups = [[setOfVecRecord[i] for i in group] for group in groups_idx if group]
#         return groups

# vibe code

import faiss
import numpy as np
from HSmodule import VectorRecord, DSU
from typing import List, Optional, Union

class FaissSearch:
    """
    Lớp FaissSearch dùng để phân cụm (clustering) một tập hợp các VectorRecord.
    
    Quy trình hoạt động:
    1. Sử dụng Faiss (HNSW hoặc IndexBinaryFlat) để tìm k-hàng xóm gần nhất (ANN) 
       cho mỗi vector trong một lô (batch).
    2. Sử dụng cấu trúc Disjoint Set Union (DSU) để gom các vector vào cùng
       một cụm nếu khoảng cách (distance) giữa chúng nhỏ hơn một ngưỡng (threshold)
       đã định.
    """

    def __init__(self, bbit: int = 8):
        """
        Khởi tạo lớp.
        :param bbit: Số bit sử dụng cho mỗi chiều khi dùng MinHash/Hamming.
        """
        self.threshold: float = 0.2  # Ngưỡng distance mặc định
        self.k_neighbors: int = 100  # k-hàng xóm gần nhất để tìm kiếm
        
        # Các thuộc tính này sẽ được thiết lập bởi các hàm setter
        self.index: Optional[Union[faiss.Index, faiss.IndexBinary]] = None
        self.metric: Optional[str] = None  # "cosine" hoặc "hamming"
        self.dim: int = 0                  # Số chiều của vector (sẽ tự động cập nhật)
        self.bbit: int = bbit

    def setDisFunc(self, metric: str):
        """
        Thiết lập metric (phương thức đo) sẽ sử dụng.
        Hàm này PHẢI được gọi trước khi chạy classify().
        
        :param metric: Tên metric ("cosine" hoặc "hamming")
        """
        if metric not in ["cosine", "hamming"]:
            raise ValueError(f"Unsupported metric: {metric}")
        self.metric = metric
        # Index sẽ được tạo sau khi biết 'dim' của dữ liệu
        self.index = None 
        self.dim = 0

    def set_threshold(self, threshold: float):
        """
        Thiết lập ngưỡng (distance) để gom 2 vector vào cùng 1 cụm.
        
        :param threshold: Ngưỡng khoảng cách.
                          - Với "cosine", đây là Cosine Distance (1 - Sim).
                          - Với "hamming", đây là Hamming Distance.
        """
        self.threshold = threshold

    def set_k_neighbors(self, k: int):
        """
        Thiết lập số lượng hàng xóm gần nhất (k) để tìm kiếm.
        :param k: Số lượng hàng xóm.
        """
        self.k_neighbors = k

    def _create_index(self, dim: int):
        """
        Hàm nội bộ để khởi tạo index Faiss dựa trên metric và 
        số chiều (dim) của dữ liệu.
        """
        self.dim = dim
        print(f"Initializing index for metric '{self.metric}' with dimension {dim}...")
        
        if self.metric == "cosine":
            # Tạo 1 đồ thị HNSW trong không gian 'dim' chiều, mỗi node có tối đa 32 cạnh
            self.index = faiss.IndexHNSWFlat(self.dim, 32)
            # Dùng Tích vô hướng. Khi vector đã L2-normalized, nó tương đương Cosine Sim.
            self.index.metric_type = faiss.METRIC_INNER_PRODUCT 
            self.index.hnsw.efSearch = 64  # Số lượng node được duyệt khi tìm kiếm
        
        elif self.metric == "hamming":
            # Với b-bit minhash → binary vector, tổng số bit = dim * bbit
            n_bits = self.dim * self.bbit
            self.index = faiss.IndexBinaryFlat(n_bits)
        
        else:
            # Lỗi này không nên xảy ra nếu đã gọi set_metric
            raise ValueError(f"Metric not set or unsupported: {self.metric}")

    def classify(self, setOfVecRecord: List[VectorRecord]) -> List[List[VectorRecord]]:
        """
        Phân cụm danh sách các VectorRecord thành các nhóm.
        
        :param setOfVecRecord: Danh sách các đối tượng VectorRecord.
        :return: Một danh sách chứa các danh sách (cụm) VectorRecord.
        """
        if not setOfVecRecord:
            return []
        
        n = len(setOfVecRecord)
        current_dim = len(setOfVecRecord[0].vec)

        if self.metric is None:
            raise ValueError("Metric not set. Call set_metric(metric) before classify().")

        # Nếu index chưa được tạo, hoặc số chiều dữ liệu thay đổi -> tạo index mới
        if self.index is None or self.dim != current_dim:
            self._create_index(current_dim)

        # Phải reset index trước mỗi lần classify để xóa dữ liệu cũ
        self.index.reset()
        
        # Đảm bảo k không lớn hơn số lượng vector
        k = min(self.k_neighbors, n)
        
        dists = None  # Khoảng cách
        idxs = None   # Chỉ số (index) của các hàng xóm

        # --- Bước 1 & 2: Chuẩn bị dữ liệu, thêm vào Index và Tìm kiếm ---
        
        if self.metric == "cosine":
            # 1. Chuẩn bị dữ liệu: stack thành mảng numpy và chuẩn hóa L2
            vecs = np.stack([v.vec for v in setOfVecRecord]).astype("float32")
            faiss.normalize_L2(vecs)  # Bắt buộc để Inner Product = Cosine Similarity
            
            # 2. Thêm vào index và tìm kiếm
            self.index.add(vecs)
            sims, idxs = self.index.search(vecs, k)
            # 3. Chuyển đổi Similarity (độ tương đồng) thành Distance (khoảng cách)
            # Cosine Distance = 1.0 - Cosine Similarity
            dists = (1.0 - sims) / 2
            print(dists)
            print("_________________________")

        elif self.metric == "hamming":
            # 1. Chuẩn bị dữ liệu: Chuyển đổi thành mảng binary
            # Chỉ lấy 8 bit cuối mỗi số, chuyển trực tiếp thành uint8
            vecs_list = [v.vec for v in setOfVecRecord]
            bin_array = np.stack([
                np.array([x % 256 for x in v], dtype=np.uint8) for v in vecs_list
            ])
            
            # 2. Thêm vào index và tìm kiếm
            self.index.add(bin_array)
            # dists trả về từ IndexBinaryFlat đã là khoảng cách Hamming
            dists, idxs = self.index.search(bin_array, k)
            

        # --- Bước 3: Gom cụm bằng Disjoint Set Union (DSU) ---
        
        dsu = DSU(n)
        for i in range(n):
            # Duyệt qua k hàng xóm (j) và khoảng cách (d) của vector i
            for j, d in zip(idxs[i], dists[i]):
                if i == j or j == -1:  # Bỏ qua chính nó hoặc index rỗng (-1)
                    continue
                
                # Nếu khoảng cách đủ nhỏ (dưới ngưỡng), gộp 2 cụm
                if d <= self.threshold:
                    dsu.unionSet(i, j)

        # --- Bước 4: Lấy kết quả các cụm ---
        
        groups_idx = dsu.getGroups()
        # Chuyển đổi danh sách chỉ số (index) về lại danh sách VectorRecord
        groups = [[setOfVecRecord[idx] for idx in group] for group in groups_idx if group]
        
        return groups
        
