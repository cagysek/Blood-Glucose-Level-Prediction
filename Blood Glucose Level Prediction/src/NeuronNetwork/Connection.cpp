//
//  Connection.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#include "Connection.hpp"

Connection::Connection()
{
    weight = getRandomWeight();
    deltaWeight = 0.0;
}

double Connection::getRandomWeight()
{
    srand((unsigned) time(NULL));
    
    return (double) rand() / RAND_MAX;
}
