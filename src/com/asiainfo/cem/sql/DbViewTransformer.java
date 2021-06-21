package com.asiainfo.cem.sql;

public interface DbViewTransformer {
    void check(DbConnection con);

    DbSchema sourceView();
    DbSchema targetView();

    String prepareSql();
}
