import urllib
import pandas as pd
import sqlalchemy as sql
from config.config import ConfigInit

class SqlConnector(ConfigInit):
    """Database connector

    Args:
        ConfigInit (object): Intialise configuration files
    """

    def __init__(self):
        """Intialise database connector
        """

        ConfigInit.__init__(self)
        self.create_connection()
        self.create_databases_tables_views()
        self.create_user()

    def create_connection(self):
        """Creates connection
        """
        params = urllib.parse.quote_plus('DRIVER='+self.db_credential['driver']+
                                         ';SERVER='+self.db_credential['server']+
                                         ';DATABASE='+self.db_credential['access_database']+
                                         ';UID='+self.db_credential['username']+
                                         ';PWD='+ self.db_credential['password'])
        self.db_credential_engine = sql.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params,
                                                      connect_args = {'autocommit':True})

        params = urllib.parse.quote_plus('DRIVER='+self.login['driver']+
                                         ';SERVER='+self.login['server']+
                                         ';DATABASE='+self.login['access_database']+
                                         ';UID='+self.login['username']+
                                         ';PWD='+ self.login['password'])
        self.engine = sql.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

    def create_databases_tables_views(self):
        """Create database, tables and views

        Args:
            queries (list): List of string that contains the query to ingest
        """

        with self.db_credential_engine.connect() as con:
            for snippet in self.dbs_tables_views_creation:
                con.execute(snippet)
        print("Databases, Tables and Views configured")

    def create_user(self):
        """Create user
        """

        with self.db_credential_engine.connect() as con:
            for snippet in self.user_creation:
                con.execute(snippet)
        print("User configured")

    def run_query(self, sql):
        """Run queries based on sql syntax

        Args:
            sql (str): Sql query

        Returns:
            Pandas Dataframe: Returns sql query wraped in pandas dataframe
        """

        return pd.read_sql(sql, con=self.engine)

    def ingest_dataframe(self, df, schema, table):
        """Ingest dataframe

        Args:
            df (Pandas Dataframe): Dataframe to ingest
            schema (str): Name of table's schema
            table (str): Name of table
        """

        df.to_sql(name=table,
                  schema=schema,
                  con = self.engine,
                  if_exists="append",
                  index=False,
                  method="multi",
                  chunksize=(2100 // len(df.columns.tolist())) - 1
                 )
        print("Data ingested")

    def ingest_query(self, queries):
        """Ingest queries

        Args:
            queries (list): List of string that contains the query to ingest
        """

        with self.engine.connect() as con:
            for query in queries:
                con.execute(query)
        print("Data ingested")