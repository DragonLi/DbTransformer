package com.asiainfo.cem.sql;

public interface DbViewTransformer {
    void check();

    DbConnectionConfiguration prepareCon();

    DbSchema sourceView();
    DbSchema targetView();

    String prepareSql();
}
