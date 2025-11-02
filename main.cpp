#include "header/MurmurHash3.hpp"
#include <cmath>
#include "../header/LSHSearch.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "../header/BloomFilter.hpp"
using namespace std;

int main()
{
    // expectedItems = 10, falsePositiveRate = 0.01
    BloomFilter bf(0, 0, 10, 0.01);

    vector<VectorRecord> input = {
        {1, {1.0, 2.0, 3.0}},
        {2, {1.0, 2.0, 3.0}}, // trùng với vector đầu
        {3, {4.0, 5.0, 6.0}},
        {4, {7.0, 8.0, 9.0}},
        {5, {4.0, 5.0, 6.0}} // trùng với vector thứ 3
    };

    // Dùng BloomFilter lọc trùng
    vector<VectorRecord> output = bf.hash(input);

    cout << "==== BloomFilter Test ====" << endl;
    cout << "Input size: " << input.size() << endl;
    cout << "Output size (unique): " << output.size() << endl;
    cout << endl;

    // In ra các vector còn lại sau khi lọc
    for (size_t i = 0; i < output.size(); ++i)
    {
        cout << "Vector " << i + 1 << ": ";
        for (double v : output[i].vec)
        {
            cout << v << " ";
        }
        cout << endl;
    }

    return 0;
}
// g++ -std=c++17 -O2 -Iheader -o my_program main.cpp source/MurmurHash3.cpp source/BloomFilter.cpp