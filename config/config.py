import yaml

class ConfigInit(object):
    """Configuration files
    """

    def __init__(self):
        """Intialise configurations
        """

        with open('config/login.yaml', 'r') as file:
            try:
                self.login = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        with open('config/db_credential.yaml', 'r') as file:
            try:
                self.db_credential = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        with open("database/dbs_tables_views_creation.sql") as file:
            self.dbs_tables_views_creation = [snippet.lstrip().rstrip().format_map(self.db_credential)
                                              for snippet in file.read().split('GO')]

        with open("database/user_creation.sql") as file:
            self.user_creation = [snippet.lstrip().rstrip().format_map(self.login)
                                  for snippet in file.read().split('GO')]