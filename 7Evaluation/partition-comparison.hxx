#pragma once
#ifndef ANDRES_PARTITION_COMPARISON_HXX
#define ANDRES_PARTITION_COMPARISON_HXX

#include <map>
#include <utility> // pair
#include <iterator> // iterator_traits
#include <cmath> // log
#include <stdexcept> // runtime_error

namespace andres {

// interface
template<class ITERATOR_0, class ITERATOR_1>
    size_t matchingPairs(ITERATOR_0, ITERATOR_0, ITERATOR_1, const bool = false);
template<class ITERATOR_0, class ITERATOR_1>
    size_t matchingPairs(ITERATOR_0, ITERATOR_0, ITERATOR_1, const bool, size_t&);
template<class ITERATOR_0, class ITERATOR_1>
    double randIndex(ITERATOR_0, ITERATOR_0, ITERATOR_1, const bool = false);
template<class ITERATOR_0, class ITERATOR_1>
    double variationOfInformation(ITERATOR_0, ITERATOR_0, ITERATOR_1, const bool = false);

// brute force code for unit tests
template<class ITERATOR_0, class ITERATOR_1>
    size_t matchingPairsBruteForce(ITERATOR_0, ITERATOR_0, ITERATOR_1, const bool = false);
template<class ITERATOR_0, class ITERATOR_1>
    size_t matchingPairsBruteForce(ITERATOR_0, ITERATOR_0, ITERATOR_1, const bool, size_t&);
template<class ITERATOR_0, class ITERATOR_1>
    double randIndexBruteForce(ITERATOR_0, ITERATOR_0, ITERATOR_1, const bool = false);

// implementation

template<class ITERATOR_0, class ITERATOR_1>
inline size_t
matchingPairsBruteForce
(
    ITERATOR_0 begin0,
    ITERATOR_0 end0,
    ITERATOR_1 begin1,
    const bool ignoreDefaultLabel
)
{
    size_t N;
    return matchingPairsBruteForce(begin0, end0, begin1, ignoreDefaultLabel, N);
}

template<class ITERATOR_0, class ITERATOR_1>
size_t
matchingPairsBruteForce
(
    ITERATOR_0 begin0,
    ITERATOR_0 end0,
    ITERATOR_1 begin1,
    const bool ignoreDefaultLabel,
    size_t& N // output: number of elements which have a non-zero label in both partitions
)
{
    typedef typename std::iterator_traits<ITERATOR_0>::value_type Label0;
    typedef typename std::iterator_traits<ITERATOR_1>::value_type Label1;

    size_t AB = 0;
    if(ignoreDefaultLabel) {
        N = 0;
        ITERATOR_1 it1 = begin1;
        for(ITERATOR_0 it0 = begin0; it0 != end0; ++it0, ++it1) {
            if(*it0 != Label0() && *it1 != Label1()) {
                ++N;
                ITERATOR_1 it1b = it1 + 1;
                for(ITERATOR_0 it0b = it0 + 1; it0b != end0; ++it0b, ++it1b) {
                    if(*it0b != Label0() && *it1b != Label1()) {
                        if((*it0 == *it0b && *it1 == *it1b) || (*it0 != *it0b && *it1 != *it1b)) {
                            ++AB;
                        }
                    }
                }
            }
        }
    }
    else {
        N = std::distance(begin0, end0);
        ITERATOR_1 it1 = begin1;
        for(ITERATOR_0 it0 = begin0; it0 != end0; ++it0, ++it1) {
            ITERATOR_1 it1b = it1 + 1;
            for(ITERATOR_0 it0b = it0 + 1; it0b != end0; ++it0b, ++it1b) {
                if( (*it0 == *it0b && *it1 == *it1b) || (*it0 != *it0b && *it1 != *it1b) ) {
                    ++AB;
                }
            }
        }

    }
    return AB;
}

template<class ITERATOR_0, class ITERATOR_1>
inline size_t
matchingPairs
(
    ITERATOR_0 begin0,
    ITERATOR_0 end0,
    ITERATOR_1 begin1,
    const bool ignoreDefaultLabel
)
{
    size_t N;
    return matchingPairs(begin0, end0, begin1, ignoreDefaultLabel, N);
}

template<class ITERATOR_0, class ITERATOR_1>
size_t
matchingPairs
(
    ITERATOR_0 begin0,
    ITERATOR_0 end0,
    ITERATOR_1 begin1,
    const bool ignoreDefaultLabel,
    size_t& N // output: number of elements which have a non-zero label in both partitions
)
{
    typedef typename std::iterator_traits<ITERATOR_0>::value_type Label0;
    typedef typename std::iterator_traits<ITERATOR_1>::value_type Label1;
    typedef std::pair<Label0, Label1> Pair;
    typedef std::map<Pair, size_t> OverlapMatrix;
    typedef std::map<Label0, size_t> RowSumMap;
    typedef std::map<Label1, size_t> ColumnSumMap;

    OverlapMatrix n;
    RowSumMap rowSum;
    ColumnSumMap columnSum;
    if(ignoreDefaultLabel) {
        N = 0;
        for(; begin0 != end0; ++begin0, ++begin1) {
            if(*begin0 != Label0() && *begin1 != Label1()) {
                ++n[Pair(*begin0, *begin1)];
                ++rowSum[*begin0];
                ++columnSum[*begin1];
                ++N;
            }
        }
    }
    else {
        N = std::distance(begin0, end0);
        for(; begin0 != end0; ++begin0, ++begin1) {
            ++n[Pair(*begin0, *begin1)];
            ++rowSum[*begin0];
            ++columnSum[*begin1];
        }
    }
    size_t A = 0.0;
    for(typename OverlapMatrix::const_iterator it = n.begin(); it != n.end(); ++it) {
        A += (it->second) * (it->second - 1);
    }
    size_t B = N * N;
    for(typename OverlapMatrix::const_iterator it = n.begin(); it != n.end(); ++it) {
        B += it->second * it->second;
    }
    for(typename RowSumMap::const_iterator it = rowSum.begin(); it != rowSum.end(); ++it) {
        B -= it->second * it->second;
    }
    for(typename ColumnSumMap::const_iterator it = columnSum.begin(); it != columnSum.end(); ++it) {
        B -= it->second * it->second;
    }
    return (A + B) / 2;
}

template<class ITERATOR_0, class ITERATOR_1>
inline double
randIndexBruteForce
(
    ITERATOR_0 begin0,
    ITERATOR_0 end0,
    ITERATOR_1 begin1,
    const bool ignoreDefaultLabel
)
{
    size_t N;
    const size_t n = matchingPairsBruteForce(begin0, end0, begin1, ignoreDefaultLabel, N);
    if(N == 0) {
        throw std::runtime_error("No element is labeled in both partitions.");
    }
    else {
        return static_cast<double>(n) * 2 / static_cast<double>(N * (N-1));
    }
}

template<class ITERATOR_0, class ITERATOR_1>
inline double
randIndex
(
    ITERATOR_0 begin0,
    ITERATOR_0 end0,
    ITERATOR_1 begin1,
    const bool ignoreDefaultLabel
)
{
    size_t N;
    const size_t n = matchingPairs(begin0, end0, begin1, ignoreDefaultLabel, N);
    if(N == 0) {
        throw std::runtime_error("No element is labeled in both partitions.");
    }
    else {
        return static_cast<double>(n) * 2 / static_cast<double>(N * (N-1));
    }
}

template<class ITERATOR_0, class ITERATOR_1>
double
variationOfInformation
(
    ITERATOR_0 begin0,
    ITERATOR_0 end0,
    ITERATOR_1 begin1,
    const bool ignoreDefaultLabel
)
{
    typedef typename std::iterator_traits<ITERATOR_0>::value_type Label0;
    typedef typename std::iterator_traits<ITERATOR_1>::value_type Label1;
    typedef std::pair<Label0, Label1> Pair;
    typedef std::map<Pair, double> PMatrix;
    typedef std::map<Label0, double> PVector0;
    typedef std::map<Label1, double> PVector1;

    // count
    size_t N = std::distance(begin0, end0);
    PMatrix pjk;
    PVector0 pj;
    PVector1 pk;
    if(ignoreDefaultLabel) {
        N = 0;
        for(; begin0 != end0; ++begin0, ++begin1) {
            if(*begin0 != Label0() && *begin1 != Label1()) {
                ++pj[*begin0];
                ++pk[*begin1];
                ++pjk[Pair(*begin0, *begin1)];
                ++N;
            }
        }
    }
    else {
        for(; begin0 != end0; ++begin0, ++begin1) {
            ++pj[*begin0];
            ++pk[*begin1];
            ++pjk[Pair(*begin0, *begin1)];
        }
    }

    // normalize
    for(typename PVector0::iterator it = pj.begin(); it != pj.end(); ++it) {
        it->second /= N;
    }
    for(typename PVector1::iterator it = pk.begin(); it != pk.end(); ++it) {
        it->second /= N;
    }
    for(typename PMatrix::iterator it = pjk.begin(); it != pjk.end(); ++it) {
        it->second /= N;
    }

    // compute information
    double H0 = 0.0;
    for(typename PVector0::const_iterator it = pj.begin(); it != pj.end(); ++it) {
        H0 -= it->second * std::log(it->second);
    }
    double H1 = 0.0;
    for(typename PVector1::const_iterator it = pk.begin(); it != pk.end(); ++it) {
        H1 -= it->second * std::log(it->second);
    }
    double I = 0.0;
    for(typename PMatrix::const_iterator it = pjk.begin(); it != pjk.end(); ++it) {
        const Label0 j = it->first.first;
        const Label1 k = it->first.second;
        const double pjk_here = it->second;
        const double pj_here = pj[j];
        const double pk_here = pk[k];
        I += pjk_here * std::log( pjk_here / (pj_here * pk_here) );
    }

    return H0 + H1 - 2.0 * I;
}

} // namespace andres

#endif // #ifndef ANDRES_PARTITION_COMPARISON_HXX
