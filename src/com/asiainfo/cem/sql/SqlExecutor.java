package com.asiainfo.cem.sql;

public class SqlExecutor {
    public void check(){
        //TODO
    }

    public void dryRun(){
        //TODO
    }

    public void runSql(DbExecEnv env,DbViewTransformer transformer){
        //TODO
        env.initialize();
        DbConnection con = env.prepareConnection(transformer.sourceView().dbConfigName);
        con.initialize();
        con.execSql(transformer.prepareSql());
    }
}
