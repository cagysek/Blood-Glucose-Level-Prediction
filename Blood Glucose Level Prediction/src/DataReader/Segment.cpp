//
//  Segment.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 26/10/2020.
//

#include "Segment.hpp"

Segment::Segment(unsigned start_id, unsigned segment_id, unsigned row_count)
{
    m_start_id = start_id;
    m_segment_id = segment_id;
    m_row_count = row_count;
}
