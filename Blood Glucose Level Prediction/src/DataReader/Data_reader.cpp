//
//  DataReader.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 18/10/2020.
//

#include "Data_reader.hpp"

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


Data_reader::Data_reader(const char *filename)
{
    m_database_file = filename;
}

unsigned Data_reader::get_input_data(std::vector<double> &input_values, unsigned limit, unsigned offset)
{
    
    input_values.clear();
    
    std::string data("CALLBACK FUNCTION");
  
    std::string sql("SELECT * FROM measuredvalue LIMIT 5;");
    
    int rc = sqlite3_exec(m_DB, sql.c_str(), callback, (void*)data.c_str(), NULL);
  
    if (rc != SQLITE_OK)
        std::cout << "Error SELECT" << std::endl;
    else {
        std::cout << "Operation OK!" << std::endl;
    }
    
    return input_values.size();
    
}

unsigned Data_reader::get_target_data(std::vector<double> &target_values, unsigned limit, unsigned offset)
{
    target_values.clear();
    
    std::string data("CALLBACK FUNCTION");
  
    std::string sql("SELECT * FROM measuredvalue;");
    
    int rc = sqlite3_exec(m_DB, sql.c_str(), callback, (void*)data.c_str(), NULL);
  
    if (rc != SQLITE_OK)
        std::cout << "Error SELECT" << std::endl;
    else {
        std::cout << "Operation OK!" << std::endl;
    }
    
    
    return target_values.size();

}

void Data_reader::open()
{
    int exit = 0;
    exit = sqlite3_open(m_database_file, &m_DB);
    
    if (exit) {
        std::cout << "Error open DB " << sqlite3_errmsg(m_DB) << std::endl;
    }
    else
    {
        std::cout << "Opened Database Successfully!" << std::endl;
  
       
    }
}

void Data_reader::close()
{
    sqlite3_close(m_DB);
}
