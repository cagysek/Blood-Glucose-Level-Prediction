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


Data_reader::Data_reader()
{
    
}

unsigned Data_reader::get_input_data(std::vector<double> &input_values, unsigned limit, unsigned offset, unsigned segment_id)
{
    input_values.clear();
    
    sqlite3_stmt *stmt;
    
    int rc;
    
    const char* sql = "SELECT * FROM measuredvalue WHERE segmentId = ? AND ID >= ? ORDER BY ID LIMIT ?;";
    rc = sqlite3_prepare_v2(m_DB, sql, -1, &stmt, NULL);
    
    rc = sqlite3_bind_int( stmt, 1, segment_id );
    rc = sqlite3_bind_int( stmt, 2, offset );
    rc = sqlite3_bind_int( stmt, 3, limit );
    
    
    if (rc != SQLITE_OK) {
        std::cout << sqlite3_errmsg(m_DB) << std::endl;
        return 0;
    }
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW)
    {
        double ist = sqlite3_column_double(stmt, 3);
        
        input_values.push_back(ist);
    }
    
    if (rc != SQLITE_DONE) {
        printf("error: ", sqlite3_errmsg(m_DB));
    }
    
    sqlite3_finalize(stmt);
    
    return input_values.size();
    
}

unsigned Data_reader::get_prediction_data(std::vector<double> &target_values, unsigned limit, unsigned offset, unsigned segment_id)
{
    return get_input_data(target_values, limit, offset, segment_id);
}

void Data_reader::init_segments(std::vector<Segment> &segments)
{
    sqlite3_stmt *stmt;
    
    const char* sql = "SELECT id, segmentid, count(*) from measuredvalue group by segmentid order by id;";
    int rc = sqlite3_prepare_v2(m_DB, sql, -1, &stmt, NULL);
    
    if (rc != SQLITE_OK) {
        std::cout << sqlite3_errmsg(m_DB) << std::endl;
        return;
    }
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW)
    {
        unsigned start_id        = sqlite3_column_int(stmt, 0);
        unsigned segment_id      = sqlite3_column_int(stmt, 1);
        unsigned row_count       = sqlite3_column_int(stmt, 2);
        
        Segment segment(start_id, segment_id, row_count);
        
        segments.push_back(segment);
    }
    
    if (rc != SQLITE_DONE) {
        printf("error: ", sqlite3_errmsg(m_DB));
    }
    
    sqlite3_finalize(stmt);
    
}


std::vector<Row> *Data_reader::get_data()
{
    sqlite3_stmt *stmt;
    
    std::vector<Row> *rows = new std::vector<Row>();
    
    const char* sql = "SELECT segmentid, ist from measuredvalue order by id;";
    int rc = sqlite3_prepare_v2(m_DB, sql, -1, &stmt, NULL);
    
    if (rc != SQLITE_OK) {
        std::cout << sqlite3_errmsg(m_DB) << std::endl;
        return rows;
    }
    
    
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW)
    {
        unsigned segment_id        = sqlite3_column_int(stmt, 0);
        double ist                 = sqlite3_column_double(stmt, 1);
        
        
        rows->push_back(Row(ist, segment_id));
    }
    
    if (rc != SQLITE_DONE) {
        printf("error: ", sqlite3_errmsg(m_DB));
    }
    
    sqlite3_finalize(stmt);
    
    return rows;
}


void Data_reader::open(const char *filename)
{
    m_database_file = filename;
    
    int result = 0;
    
    result = sqlite3_open(filename, &m_DB);
    
    if (result)
    {
        std::cout << "Databaze nenalezena " << sqlite3_errmsg(m_DB) << std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        std::cout << "Databaze pripojena!" << std::endl;
    }
}

void Data_reader::close()
{
    sqlite3_close(m_DB);
}

std::vector<double>* Data_reader::get_ini_data(const char* filename)
{
    std::vector<double> *data = new std::vector<double>();
    
    
    std::string line;
    std::ifstream input (filename);
    
    std::string delimeter = "=";
    
    
    if (input.is_open())
    {
        while ( getline (input,line) )
        {
            //std::cout << line << '\n';
            
            int pos = (int)line.find(delimeter);
            
            if (pos != std::string::npos)
            {
                std::string val_str = line.substr(pos + 1, line.length());
                
                data->push_back(std::stod(val_str));
            }
        }
    }
    else
    {
        std::cout << "Ini soubor neuronove site nebyl nalezen";
        exit(EXIT_FAILURE);
    }
    
    input.close();
    
    
    return data;
}
