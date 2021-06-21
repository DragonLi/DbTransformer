package com.asiainfo.cem.sql;

import java.util.Map;

public abstract class DbConnectionConfiguration {
    abstract Map<String,String> getParameters();
}
