//
//  Connection.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#include "Connection.hpp"
#include <iostream>
#include <random>

Connection::Connection()
{
    weight = get_random_weight();
    delta_weight = 0.0;
    transmitted_value_counter = 0.0;
    last_transmitted_value = 0.0;
    transmitted_value_relative_error_counter = 0.0;
}

double Connection::get_random_weight()
{
    // generátor náhodných čísel 0 až 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);//uniform distribution between 0 and 1
    
    return dis(gen);
}
