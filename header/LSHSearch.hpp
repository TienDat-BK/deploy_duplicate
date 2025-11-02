#pragma once
#include "STL.hpp"
#include "DSU.hpp"
#include "VectorRecord.hpp"
using namespace std;

class LSHSearch
{

public:
    // double (*disFunc)(const VectorRecord &, const VectorRecord &) = &Search::hammingDistance; // con trỏ hàm tính khoảng cách ( jarcard, hamming,... sẽ đc dùng ở các class con)
    function<double(const VectorRecord &, const VectorRecord &)> disFunc;
    int bandSize = 4;   // kích thước mỗi band
    int num_bands = -1; // số band

    double threshold = 0.4; // ngưỡng để quyết định 2 vecRecord có cùng nhóm hay không
private:
    pair<std::vector<double>::const_iterator, std::vector<double>::const_iterator> getband(const VectorRecord &, int, int); // trả về iterator của band thứ band_index

    struct bandHash
    {
        size_t operator()(vector<double>::const_iterator bandStart, vector<double>::const_iterator bandEnd) const;
    };

    struct pairHash
    {
        size_t operator()(const pair<int, int> &p) const;
    };

public:
    /*
    setOfVecRecord (input)
    └─> Hash band → setOfBucket
          └─> DSU → groups (index)
                 └─> move setOfVecRecord[idx] vào groupVecRecords
                        └─> move groupVecRecords vào result
    */
    vector<vector<VectorRecord>> classifyByBand(const vector<VectorRecord> &);

public:
    static double jarcardDistance(const VectorRecord &, const VectorRecord &); // MinHash
    static double hammingDistance(const VectorRecord &, const VectorRecord &); // SimHash

    void setDisFunc(string nameDisFunc)
    {
        if (nameDisFunc == "jarcard")
            this->disFunc = &LSHSearch::jarcardDistance;
        else if (nameDisFunc == "hamming")
            this->disFunc = &LSHSearch::hammingDistance;
        else
            throw invalid_argument("Invalid distance function name");
    }

    vector<vector<VectorRecord>> classify(const vector<VectorRecord> &setOfVec);
    // vector<vector<VectorRecord>> classify(vector<vector<double>>);
};
