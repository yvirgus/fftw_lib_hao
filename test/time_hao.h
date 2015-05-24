#ifndef TIME_HAO_H
#define TIME_HAO_H

#include <time.h>

class Timer_hao
{
   public:
    double seconds;
    int    timing_flag;
    time_t timerinit,timerend;
    Timer_hao();
    Timer_hao(double);
    Timer_hao(const Timer_hao& );
    ~Timer_hao();
   
    Timer_hao& operator  = (const Timer_hao&);
   
    void init();
    void end();
    void print_init() const;
    void print_end() const;
    void print_accumulation() const;
};

struct tm second_to_tm(double);
#endif

