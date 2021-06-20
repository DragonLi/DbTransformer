package com.asiainfo.cem.sql;

public interface DbViewTransformer {
    void check();
    void dryRun();

    DbSchema sourceView();
    DbSchema targetView();

    String prepareSql();

}
