from raghub_core.storage.local_sql import SQLStorage


class MySQLStorageExt(SQLStorage):
    """
    MySQL storage extension for Raghub.
    """

    name = "mysql"
