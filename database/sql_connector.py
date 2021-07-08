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
        self.create_database()
        self.create_tables()
        self.create_views()
        self.create_user()

    def create_connection(self):
        """Creates connection
        """

        params = urllib.parse.quote_plus('DRIVER='+self.sql_credentials['driver']+
                                         ';SERVER='+self.sql_credentials['server']+
                                         ';DATABASE='+self.sql_credentials['db_admin']['default_database']+
                                         ';UID='+self.sql_credentials['db_admin']['username']+
                                         ';PWD='+ self.sql_credentials['db_admin']['password'])
        self.db_admin_credential_engine = sql.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params,
                                                      connect_args = {'autocommit':True})

        params = urllib.parse.quote_plus('DRIVER='+self.sql_credentials['driver']+
                                         ';SERVER='+self.sql_credentials['server']+
                                         ';DATABASE='+self.sql_credentials['database']+
                                         ';UID='+self.sql_credentials['db_user']['username']+
                                         ';PWD='+ self.sql_credentials['db_user']['password'])
        self.db_user_engine = sql.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

    def create_database(self):
        """Create databases
        """

        with self.db_admin_credential_engine.connect() as con:
            for snippet in self.db_creation:
                con.execute(snippet)
        print("Databases configured")

    def create_tables(self):
        """Create tables
        """

        with self.db_admin_credential_engine.connect() as con:
            for snippet in self.table_creation:
                con.execute(snippet)
        print("Tables configured")

    def create_views(self):
        """Create views
        """

        with self.db_admin_credential_engine.connect() as con:
            for snippet in self.view_creation:
                con.execute(snippet)
        print("Views configured")

    def create_user(self):
        """Create user
        """

        with self.db_admin_credential_engine.connect() as con:
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

        return pd.read_sql(sql, con=self.db_user_engine)

    def ingest_dataframe(self, df, schema, table):
        """Ingest dataframe

        Args:
            df (Pandas Dataframe): Dataframe to ingest
            schema (str): Name of table's schema
            table (str): Name of table
        """

        df.to_sql(name=table,
                  schema=schema,
                  con = self.db_user_engine,
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

        with self.db_user_engine.connect() as con:
            for query in queries:
                con.execute(query)
        print("Data ingested")