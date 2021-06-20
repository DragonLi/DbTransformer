package com.asiainfo.cem.sql;

import java.util.List;

public abstract class SimpleDbViewTransformer implements DbViewTransformer {

    private final DbSchema sourceSchema;
    private final DbSchema targetSchema;

    abstract String prepareSelectExpr(String schemaName);
    abstract DbViewTransformer prepareFrom();
    abstract String prepareWhere();
    abstract String prepareGroupBy();
    abstract String prepareOrderBy();
    abstract Integer prepareLimit();
    abstract Integer prepareOffset();

    @Override
    public DbSchema sourceView() {
        return sourceSchema;
    }

    @Override
    public DbSchema targetView() {
        return targetSchema;
    }

    public SimpleDbViewTransformer(DbSchema src, DbSchema target) {
        this.sourceSchema = src;
        this.targetSchema =target;
    }

    @Override
    public String prepareSql() {
        //TODO required check
        StringBuilder builder = new StringBuilder();
        builder.append("select ");
        List<String> fieldNames = sourceSchema.FieldNames;
        for (int i = 0, l = fieldNames.size(); i < l; i++) {
            String fieldName = fieldNames.get(i);
            builder.append(prepareSelectExpr(fieldName));
            builder.append(" as ").append(fieldName);
            if (i != l-1){
                builder.append(",");
            }
        }
        builder.append(" from ").append(prepareFrom().prepareSql());

        //optional
        String t;
        if ((t = prepareWhere()) != null){
            builder.append(" where ").append(t);
        }
        if ((t = prepareGroupBy()) != null){
            builder.append(" group by ").append(t);
        }
        if ((t = prepareOrderBy()) != null){
            builder.append(" order by ").append(t);
        }
        Integer i;
        if ((i = prepareLimit()) != null){
            builder.append(" limit ").append(i);
            if ((i = prepareOffset()) != null){
                builder.append(" offset ").append(i);
            }
        }
        return builder.toString();
    }
}
