#include "../header/MinHash.hpp"

// sig length = 200

size_t mini_hash(const string &in)
{
    std::hash<string> ha;
    return ha(in);
}

uint64_t hash64(uint64_t x, uint64_t seed)
{
    x ^= seed;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

VectorRecord MinHash::hash_1(const VectorRecord &vec)
{
    const size_t k = this->outputDim;
    vector<uint64_t> sig(k, UINT64_MAX);

    // Convert double -> uint64_t
    vector<uint64_t> vals(vec.vec.size());
    for (size_t i = 0; i < vec.vec.size(); i++)
        vals[i] = (uint64_t)(vec.vec[i]); // safe conversion

    for (size_t seed = 0; seed < k; seed++)
    {
        for (uint64_t x : vals)
        {
            uint64_t h = hash64(x, seed);
            sig[seed] = min(sig[seed], h);
        }
    }

    // Convert to double if cần
    vector<double> rt(sig.begin(), sig.end());
    return VectorRecord(vec.id, rt);
}

vector<VectorRecord> MinHash::hash(const vector<VectorRecord> &input)
{
    int nums = input.size();
    vector<VectorRecord> rt(nums);
    for (int i = 0; i < nums; ++i)
    {
        rt[i] = this->hash_1(input[i]);
    }
    return rt;
}
