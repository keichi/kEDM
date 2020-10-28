#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <chrono>

class Timer
{
private:
    bool _is_running;
    std::chrono::steady_clock::time_point _start;
    std::chrono::steady_clock::duration _total;

    double _to_millis(std::chrono::steady_clock::duration duration) const
    {
        std::chrono::duration<double, std::milli> duration_ms(duration);
        return duration_ms.count();
    }

public:
    Timer()
        : _is_running(false),
          _total(std::chrono::steady_clock::duration::zero())
    {
    }

    void start()
    {
        _is_running = true;
        _start = std::chrono::steady_clock::now();
    }

    double stop()
    {
        _is_running = false;

        const auto e = std::chrono::steady_clock::now() - _start;

        _total += e;

        return _to_millis(e);
    }

    void reset()
    {
        _is_running = false;
        _total = std::chrono::steady_clock::duration::zero();
    }

    bool is_running() const { return _is_running; }

    double elapsed() const { return _to_millis(_total); }
};

#endif
