//
//  DataReader.hpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 18/10/2020.
//

#ifndef DataReader_hpp
#define DataReader_hpp

#include <stdio.h>
#include <string>
#include <iostream>
#include <sqlite3.h>
#include <vector>
#include "Segment.hpp"
#include "Row.hpp"
#include <fstream>



class Data_reader
{
    public:
        Data_reader();
        void close();
        void open(const char *filename);
        
        // vrací počet položek -> běží v cyklu, takže jetli vrátí 0 -> konec
        unsigned get_input_data(std::vector<double> &input_values, unsigned limit, unsigned offset, unsigned segmentId);
        unsigned get_prediction_data(std::vector<double> &target_values, unsigned limit, unsigned offset, unsigned segmentId);
        void init_segments(std::vector<Segment> &segments);
        
        std::vector<Row> *get_data();
        
        std::vector<double> *get_ini_data(const char *filename);
    
    private:
        const char *m_database_file;
        sqlite3* m_DB;
};


#endif /* DataReader_hpp */
