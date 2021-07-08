import yaml

class ConfigInit(object):
    """Configuration files
    """

    def __init__(self):
        """Intialise configurations
        """

        with open('config/sql_credential.yaml', 'r') as file:
            try:
                self.sql_credentials = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        with open("database/dbs.sql") as file:
            self.db_creation = [snippet.lstrip().rstrip().format_map(self.sql_credentials)
                                for snippet in file.read().split('GO')]

        with open("database/tables.sql") as file:
            self.table_creation = [snippet.lstrip().rstrip().format_map(self.sql_credentials)
                                   for snippet in file.read().split('GO')]

        with open("database/views.sql") as file:
            self.view_creation = [snippet.lstrip().rstrip().format_map(self.sql_credentials)
                                  for snippet in file.read().split('GO')]

        with open("database/user_creation.sql") as file:
            self.user_creation = [snippet.lstrip().rstrip().format_map(
                                                                {**self.sql_credentials['db_user'],
                                                                 'database':self.sql_credentials['database'],
                                                                 'schema':self.sql_credentials['schema']
                                                                 })
                                  for snippet in file.read().split('GO')]