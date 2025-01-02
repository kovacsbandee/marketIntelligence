import os
import logging
from typing import List, Dict, Type
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, Session

# Load environment variables from .env file
load_dotenv()


# rows = [{"time": "2025-01-01 10:00:00", "company_id": 1, "open": 145.00, "high": 146.00, "low": 144.50, "close": 145.50, "volume": 1000000},
#         {"time": "2025-01-01 10:01:00", "company_id": 1, "open": 145.50, "high": 146.20, "low": 144.80, "close": 146.00, "volume": 800000},]


class PostgresAdapter:
    """
    A utility class for interacting with a PostgreSQL database using SQLAlchemy.

    Attributes:
        db_host (str): Database host address.
        db_name (str): Database name.
        db_user (str): Database username.
        db_password (str): Database password.
        db_port (str): Database port.
        engine (Engine): SQLAlchemy engine for database connections.
        logger (Logger): Logger for logging messages.
    """

    def __init__(
        self, db_host=None, db_name=None, db_user=None, db_password=None, db_port=None
    ):
        """
        Initialize the database adapter.

        Args:
            db_host (str): Optional database host address. Defaults to environment variable DB_HOST.
            db_name (str): Optional database name. Defaults to environment variable DB_NAME.
            db_user (str): Optional database username. Defaults to environment variable DB_USER.
            db_password (str): Optional database password. Defaults to environment variable DB_PASSWORD.
            db_port (str): Optional database port. Defaults to environment variable DB_PORT.
        """
        self.db_host = db_host or os.getenv("DB_HOST")
        self.db_name = db_name or os.getenv("DB_NAME")
        self.db_user = db_user or os.getenv("DB_USER")
        self.db_password = db_password or os.getenv("DB_PASSWORD")
        self.db_port = db_port or os.getenv("DB_PORT")

        if not all(
            [self.db_host, self.db_name, self.db_user, self.db_password, self.db_port]
        ):
            raise ValueError(
                "Database connection parameters are incomplete. Check your environment or arguments."
            )

        self.engine = create_engine(
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        )
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def provide_session(self) -> Session:
        """
        Provide a new database session.

        Returns:
            Session: A new SQLAlchemy session object.
        """
        base_session = sessionmaker(bind=self.engine)
        return base_session()

    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope for database operations.

        Yields:
            Session: A new session within a transactional scope.
        """
        session = self.provide_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Transaction failed: {e}")
            raise
        finally:
            session.close()

    def insert_data(self, table: Type, rows: List[Dict]):
        """
        Insert multiple rows into a table.

        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.
            rows (List[Dict]): A list of dictionaries representing rows to insert.
        """
        with self.session_scope() as session:
            session.bulk_insert_mappings(table, rows)
            self.logger.info(f"Inserted data into {table.__tablename__} successfully.")

    def verify_table(self, table: Type, limit: int = 10, filters: Dict = None):
        """
        Verify the contents of a table by querying rows.

        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.
            limit (int): Maximum number of rows to fetch. Defaults to 10.
            filters (Dict): Optional filters as column-value pairs.
        """
        with self.session_scope() as session:
            query = session.query(table)
            if filters:
                for column, value in filters.items():
                    query = query.filter(getattr(table, column) == value)
            rows = query.limit(limit).all()

            if not rows:
                self.logger.info(f"The table {table.__tablename__} is empty.")
                return

            inspector = inspect(table)
            columns = [column.key for column in inspector.mapper.columns]
            for row in rows:
                row_data = {column: getattr(row, column) for column in columns}
                print(row_data)

    def list_tables(self):
        """
        List all table names in the connected database.

        Returns:
            List[str]: A list of table names.
        """
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def create_table(self, table: Type):
        """
        Create a table if it does not exist.

        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.
        """
        table.__table__.create(bind=self.engine, checkfirst=True)
        self.logger.info(f"Table {table.__tablename__} created.")

    def drop_table(self, table: Type):
        """
        Drop a table if it exists.

        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.
        """
        table.__table__.drop(bind=self.engine, checkfirst=True)
        self.logger.info(f"Table {table.__tablename__} dropped.")

    def fetch_all(self, table: Type, limit: int = None) -> List[Dict]:
        """
        Fetch all rows from a table, optionally limiting the number of rows.

        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.
            limit (int): Optional maximum number of rows to fetch.

        Returns:
            List[Dict]: A list of rows as dictionaries.
        """
        with self.session_scope() as session:
            query = session.query(table)
            if limit:
                query = query.limit(limit)
            return query.all()

    def fetch_filtered(self, table: Type, filters: Dict) -> List[Dict]:
        """
        Fetch rows from a table based on filter criteria.

        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.
            filters (Dict): Filters as column-value pairs.

        Returns:
            List[Dict]: A list of rows as dictionaries.
        """
        with self.session_scope() as session:
            query = session.query(table)
            for column, value in filters.items():
                query = query.filter(getattr(table, column) == value)
            return query.all()
