package com.asiainfo.cem.sql;

import java.util.List;

public class DbSchema {
    public final boolean isPersistent;
    public final List<java.lang.String> FieldNames;
    //TODO Schema的名字不能重复
    public final String name;
    public final String dbConfigName;

    public DbSchema(boolean isPersistent, List<java.lang.String> fieldNames, String name, String dbConfigName) {
        this.isPersistent = isPersistent;
        FieldNames = fieldNames;
        this.name = name;
        this.dbConfigName = dbConfigName;
    }
}
