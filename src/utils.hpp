#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>

#include "types.hpp"

namespace edm
{

std::ostream &operator<<(std::ostream &os, const SimplexLUT &lut);

template <class T> class Counter
{
private:
    T i_;

public:
    using difference_type = T;
    using value_type = T;
    using pointer = T;
    using reference = T &;
    using iterator_category = std::input_iterator_tag;

    explicit Counter(T i) : i_(i) {}
    T operator*() const noexcept { return i_; }
    Counter &operator++() noexcept
    {
        i_++;
        return *this;
    }
    bool operator==(const Counter &rhs) const { return i_ == rhs.i_; }
    bool operator!=(const Counter &rhs) const { return i_ != rhs.i_; }
};


}

#endif
