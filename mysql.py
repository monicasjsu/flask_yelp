import json
import sqlalchemy as db


class Engine:
    instance = None

    @staticmethod
    def get_db_conn():
        """ Static access method. """
        if Engine.instance is None:
            Engine()
        return Engine.instance

    def __init__(self):
        """ Virtually private constructor. """
        if Engine.instance is not None:
            raise Exception("Singleton not defined")
        else:
            Engine.instance = self.__get_db_engine()

    @staticmethod
    def __get_db_engine():
        with open('db_config.json', 'r') as data:
            config = json.load(data)
            conn = "mysql://{}:{}@{}/{}".format(config['db_username'], config['db_password'],
                                                config['rds_url'], config['db_name'])
            return db.create_engine(conn)
