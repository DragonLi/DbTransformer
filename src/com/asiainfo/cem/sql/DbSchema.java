package com.asiainfo.cem.sql;

import java.util.ArrayList;
import java.util.List;

public class DbSchema {
    public final boolean IsPersistent;
    public final List<String> FieldNames;
    public final String name;

    public DbSchema() {
        FieldNames = new ArrayList<>();
        IsPersistent = false;
        name = null;
    }
}
