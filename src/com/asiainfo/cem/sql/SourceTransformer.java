package com.asiainfo.cem.sql;

public class SourceTransformer implements DbViewTransformer {

    private final DbSchema targetSchema;

    public SourceTransformer(DbSchema targetSchema) {
        this.targetSchema = targetSchema;
    }

    @Override
    public void check(DbConnection con) {
        con.checkSchema(targetSchema);
    }

    @Override
    public String prepareCon() {
        return null;
    }

    @Override
    public DbSchema sourceView() {
        return null;
    }

    @Override
    public DbSchema targetView() {
        return targetSchema;
    }

    @Override
    public java.lang.String prepareSql() {
        return null;
    }
}
