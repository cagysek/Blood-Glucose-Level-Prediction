//
//  Segment.hpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 26/10/2020.
//

#ifndef Segment_hpp
#define Segment_hpp

#include <stdio.h>

class Segment
{
    public:
        Segment(unsigned start_id, unsigned segment_id, unsigned row_count);
        unsigned m_start_id;
        unsigned m_segment_id;
        unsigned m_row_count;
    
};

#endif /* Segment_hpp */
