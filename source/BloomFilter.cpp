#include "../header/BloomFilter.hpp"

BloomFilter::BloomFilter(int i, int o, size_t expectedItems, double falsePositiveRate) : IHash(i, o)
{
    // Tính toán kích thước m và số hàm băm k dựa trên công thức Bloom filter
    m = static_cast<size_t>(-(expectedItems * log(falsePositiveRate)) / (log(2) * log(2)));
    k = static_cast<int>((m / expectedItems) * log(2));
    bitArray.resize(m, false);
}

vector<size_t> BloomFilter::getHashIndices(const VectorRecord &rec, uint64_t &fingerprint) const
{
    uint64_t out[2];
    MurmurHash3_x64_128(
        rec.vec.data(),
        (int)(rec.vec.size() * sizeof(double)),
        0,
        &out);

    uint64_t h1 = out[0];
    uint64_t h2 = out[1];
    if (h2 % m == 0)
        h2++; // tránh trường hợp h2 = 0

    vector<size_t> indices(k);
    for (int i = 0; i < k; i++)
    {
        indices[i] = (h1 + i * h2) % m;
    }

    fingerprint = h1; // dùng h1 làm fingerprint
    return indices;
}

bool BloomFilter::possiblyContains(vector<size_t> &indices) const
{
    for (size_t idx : indices)
    {
        if (!bitArray[idx])
            return false;
    }
    return true;
}

vector<VectorRecord> BloomFilter::hash(const vector<VectorRecord> &input)
{
    vector<VectorRecord> unique;
    unique.reserve(input.size());
    unordered_set<uint64_t> realHashes;

    for (const auto &vec : input)
    {
        uint64_t fp;
        vector<size_t> indices = getHashIndices(vec, fp);

        if (!possiblyContains(indices))
        {
            for (size_t idx : indices)
                bitArray[idx] = true;
            unique.push_back(vec);
            realHashes.insert(fp);
        }
        else
        {
            if (realHashes.find(fp) == realHashes.end())
            {
                unique.push_back(vec);
                realHashes.insert(fp);
            }
        }
    }

    return unique;
}
