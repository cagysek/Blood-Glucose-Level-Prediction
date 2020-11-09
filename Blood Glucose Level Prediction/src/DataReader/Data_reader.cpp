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

void Data_reader::open(const char *filename)
{
    m_database_file = filename;
    
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
