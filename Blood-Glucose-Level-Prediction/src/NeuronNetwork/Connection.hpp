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

class Connection
{
    public:
        Connection();
        double weight;
        double delta_weight;
        
        // počítadlo hodnot, které hrana předala
        double transmitted_value_counter;
    
        // počítadlo hodnot, které hrana předala a výsledná relativní chyba byla pod 15%
        double transmitted_value_relative_error_counter;
    
        // pomocná proměnná pro držení hodnoty, pro zpětné přičtení do counteru
        double last_transmitted_value;
    
    private:
        double get_random_weight();
    
};

#endif /* Connection_hpp */
