//
//  Row.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 10/11/2020.
//

#include "Row.hpp"


Row::Row(double ist, unsigned segment_id)
{
    m_ist = ist;
    m_segment_id = segment_id;
}
