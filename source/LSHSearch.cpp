#include "../header/LSHSearch.hpp"

double LSHSearch::hammingDistance(const VectorRecord &vec1, const VectorRecord &vec2)
{

    auto &a = vec1.vec;
    auto &b = vec2.vec;
    int dim = a.size();
    double epsilon = 1e-9;
    double dis = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (fabs(a[i] - b[i]) > epsilon)
            dis = dis + 1;
    }
    return dis / dim;
}

double LSHSearch::jarcardDistance(const VectorRecord &vec1, const VectorRecord &vec2)
{

    set<double> s1(vec1.vec.begin(), vec1.vec.end());
    set<double> s2(vec2.vec.begin(), vec2.vec.end());

    vector<double> inter;
    set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(), back_inserter(inter));

    vector<double> uni;
    set_union(s1.begin(), s1.end(), s2.begin(), s2.end(), back_inserter(uni));

    return 1 - (double)inter.size() / uni.size();
}

pair<std::vector<double>::const_iterator, std::vector<double>::const_iterator> LSHSearch::getband(const VectorRecord &vec, int band_index, int band_size)
{
    // trả về iterator của band thứ band_index

    size_t start = band_index * band_size;
    size_t end = start + band_size;
    // nếu band cuối cùng không đủ band_size thì lấy hết phần còn lại
    if (end > vec.vec.size())
        end = vec.vec.size();

    // nếu start vượt quá kích thước vector thì throw lỗi
    if (start >= vec.vec.size())
        throw invalid_argument("Band index out of range");

    if (end > vec.vec.size())
        end = vec.vec.size();

    return {vec.vec.begin() + start, vec.vec.begin() + end};
}

size_t LSHSearch::bandHash::operator()(std::vector<double>::const_iterator bandStart, std::vector<double>::const_iterator bandEnd) const
{
    size_t seed = bandEnd - bandStart;
    for (auto it = bandStart; it != bandEnd; ++it)
    {
        double val = *it;
        seed ^= std::hash<double>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

size_t LSHSearch::pairHash::operator()(const pair<int, int> &p) const
{
    size_t seed = 0;
    seed ^= std::hash<int>()(p.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>()(p.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

vector<vector<VectorRecord>> LSHSearch::classifyByBand(const vector<VectorRecord> &setOfVecRecord)
{
    int dim = setOfVecRecord[0].vec.size();
    cout << "Classify by band with " << setOfVecRecord.size() << " vectors, each of dimension " << setOfVecRecord[0].vec.size() << ", using " << this->num_bands << " bands." << endl;

    this->num_bands = dim / bandSize + (dim % bandSize != 0 ? 1 : 0);

    // kiểm tra tính hợp lệ của tham số
    if (this->num_bands > dim)
        throw invalid_argument("Number of bands exceeds vector dimension");
    // kiểm tra hàm khoảng cách đã được gán chưa
    if (this->disFunc == nullptr)
        throw invalid_argument("Distance function is not set");

    // ở đây ta sẽ lấy key: size_t là giá trị sau khi hash của Band để giảm bộ nhows xuống
    vector<unordered_map<size_t, vector<int>>> setOfBucket(this->num_bands);
    LSHSearch::bandHash bandHasher;

    // STEP 1 : Băm các Band và đánh dấu các VecRecord chung Band
    int vecCnt = 0;
    for (const VectorRecord &vecRecord : setOfVecRecord)
    {
        for (int i = 0; i < this->num_bands; i++)
        {
            auto [bandStart, bandEnd] = this->getband(vecRecord, i, dim / this->num_bands);
            auto valueBandHash = bandHasher(bandStart, bandEnd);
            setOfBucket[i][valueBandHash].push_back(vecCnt);
        }
        vecCnt++;
    }

    // STEP 2 : Dùng DSU để nhóm các VecRecord chung Band
    DSU dsu(vecCnt);
    for (auto &bucket : setOfBucket)
    {
        for (auto &[bandHashValue, setOfVectorInSame] : bucket)
        {
            // lấy vecRecord đầu tiên làm seed
            int u = setOfVectorInSame[0];
            while (setOfVectorInSame.size() > 1)
            {
                for (int i = setOfVectorInSame.size() - 1; i > 0; i--)
                {
                    // tinhk toán khoảng cách giữa các vecRecord trong cùng bucket
                    // nếu khoảng cách nhỏ hơn threshold thì gộp chúng lại
                    int v = setOfVectorInSame[i];
                    if (this->disFunc(setOfVecRecord[u], setOfVecRecord[v]) < this->threshold)
                    {
                        dsu.unionSet(u, v);
                        // xóa khỏi setOfVectorInSame để tránh tính toán lại
                        swap(setOfVectorInSame[i], setOfVectorInSame.back());
                        setOfVectorInSame.pop_back();
                    }
                }
                // xóa vecRecord đầu tiên khỏi setOfVectorInSame để tránh tính toán lại
                swap(setOfVectorInSame[0], setOfVectorInSame.back());
                setOfVectorInSame.pop_back();
            }
        }
    }
    vector<vector<int>> groups = dsu.getGroups();

    // STEP 3 : Tạo kết quả trả về
    vector<vector<VectorRecord>> result;
    for (const auto &group : groups)
    {
        vector<VectorRecord> groupVecRecords;
        for (int idx : group)
        {
            groupVecRecords.push_back(move(setOfVecRecord[idx]));
        }
        result.push_back(move(groupVecRecords));
    }
    return result;
}

vector<vector<VectorRecord>> LSHSearch::classify(const vector<VectorRecord> &setOfVec)
{
    // Sử dụng hàm classifyByBand để phân loại
    return classifyByBand(setOfVec);
}
