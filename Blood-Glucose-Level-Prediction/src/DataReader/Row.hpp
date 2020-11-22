//
//  Row.hpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 10/11/2020.
//

#ifndef Row_hpp
#define Row_hpp

#include <stdio.h>

class Row
{
    public:
        Row(double m_ist, unsigned m_segment_id);
        double get_ist() { return m_ist; }
        unsigned get_segment_id() { return m_segment_id; }
    
    private:
        double m_ist;
        unsigned m_segment_id;
};

#endif /* Row_hpp */
