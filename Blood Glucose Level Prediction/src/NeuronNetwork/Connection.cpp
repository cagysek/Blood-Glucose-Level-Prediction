//
//  Connection.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#include "Connection.hpp"

Connection::Connection()
{
    weight = get_random_weight();
    delta_weight = 0.0;
}

double Connection::get_random_weight()
{
    srand((unsigned) time(NULL));
    
    return (double) rand() / RAND_MAX;
}
