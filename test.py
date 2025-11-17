from source.Preprocessor import *
if __name__ == "__main__":
    import time

    # Test nhỏ
    texts = ["hello world " * 1000] * 200  # ~50k ký tự
    sh = Shingling(k=5)

    start = time.time()
    sh(texts)
    print(f"Nhỏ: {time.time() - start:.3f}s")

    # Test lớn
    start = time.time()
    texts = ["hello world " * 1000] * 20000  # ~2MB
    start = time.time()
    sh(texts)
    print(f"Lớn: {time.time() - start:.3f}s")