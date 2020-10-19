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


class DataReader
{
    public:
        DataReader(const char *filename);
    void close();
    void open();
    
    // vrací počet položek -> běží v cyklu, takže jetli vrátí 0 -> konec
    unsigned getInputData(std::vector<double> &inputValues, unsigned limit, unsigned offset);
    unsigned getTargetData(std::vector<double> &targetValues, unsigned limit, unsigned offset);
    
    
    private:
        const char *databaseFile;
        sqlite3* DB;
};


#endif /* DataReader_hpp */
