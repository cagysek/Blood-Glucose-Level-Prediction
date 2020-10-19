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


class Data_reader
{
    public:
        Data_reader(const char *filename);
        void close();
        void open();
        
        // vrací počet položek -> běží v cyklu, takže jetli vrátí 0 -> konec
        unsigned get_input_data(std::vector<double> &input_values, unsigned limit, unsigned offset);
        unsigned get_target_data(std::vector<double> &target_values, unsigned limit, unsigned offset);
        
    
    private:
        const char *m_database_file;
        sqlite3* m_DB;
};


#endif /* DataReader_hpp */
