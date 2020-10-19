//
//  DataReader.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 18/10/2020.
//

#include "DataReader.hpp"

static int callback(void* data, int argc, char** argv, char** azColName)
{
    int i;
    fprintf(stderr, "%s: ", (const char*)data);
  
    for (i = 0; i < argc; i++) {
        printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
    }
  
    printf("\n");
    return 0;
}


DataReader::DataReader(const char *filename)
{
    databaseFile = filename;
}

unsigned DataReader::getInputData(std::vector<double> &inputValues, unsigned limit, unsigned offset)
{
    
    inputValues.clear();
    
    std::string data("CALLBACK FUNCTION");
  
    std::string sql("SELECT * FROM measuredvalue LIMIT 5;");
    
    int rc = sqlite3_exec(DB, sql.c_str(), callback, (void*)data.c_str(), NULL);
  
    if (rc != SQLITE_OK)
        std::cout << "Error SELECT" << std::endl;
    else {
        std::cout << "Operation OK!" << std::endl;
    }
    
    return inputValues.size();
    
}

unsigned DataReader::getTargetData(std::vector<double> &targetValues, unsigned limit, unsigned offset)
{
    targetValues.clear();
    
    std::string data("CALLBACK FUNCTION");
  
    std::string sql("SELECT * FROM measuredvalue;");
    
    int rc = sqlite3_exec(DB, sql.c_str(), callback, (void*)data.c_str(), NULL);
  
    if (rc != SQLITE_OK)
        std::cout << "Error SELECT" << std::endl;
    else {
        std::cout << "Operation OK!" << std::endl;
    }
    
    
    return targetValues.size();

}

void DataReader::open()
{
    int exit = 0;
    exit = sqlite3_open(databaseFile, &DB);
    
    if (exit) {
        std::cout << "Error open DB " << sqlite3_errmsg(DB) << std::endl;
    }
    else
    {
        std::cout << "Opened Database Successfully!" << std::endl;
  
       
    }
}

void DataReader::close()
{
    sqlite3_close(DB);
}
