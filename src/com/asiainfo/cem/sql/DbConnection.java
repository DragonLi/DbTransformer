package com.asiainfo.cem.sql;

public interface DbConnection {
    void initialize();
    void execSql(String sql);
}
