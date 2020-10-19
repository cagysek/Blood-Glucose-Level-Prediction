//
//  Connection.hpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#ifndef Connection_hpp
#define Connection_hpp

#include <stdio.h>
#include <cstdlib>
#include <time.h>

class Connection
{
    public:
        Connection();
        double weight;
        double delta_weight;
    
    private:
        double get_random_weight();
    
};

#endif /* Connection_hpp */
