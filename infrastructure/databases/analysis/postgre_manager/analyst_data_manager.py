import os
import logging
from typing import List, Dict, Type
from contextlib import contextmanager

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, func, Integer, BigInteger, tuple_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import insert


# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Singleton holder for the manager
_HANDLER_SINGLETON = None

def get_analyst_data_handler():
    """
    Returns a process-wide singleton AnalystDataManager to reuse engine/session.
    """
    global _HANDLER_SINGLETON
    if _HANDLER_SINGLETON is None:
        _HANDLER_SINGLETON = AnalystDataManager()
        logger.info("AnalystDataManager initialized with database connection.")
    return _HANDLER_SINGLETON


class PostgresManager:
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
        self.db_host = db_host or os.getenv("ANALYST_DB_HOST")
        self.db_name = db_name or os.getenv("ANALYST_DB_NAME")
        self.db_user = db_user or os.getenv("ANALYST_DB_USER")
        self.db_password = db_password or os.getenv("ANALYST_DB_PASSWORD")
        self.db_port = db_port or os.getenv("ANALYST_DB_PORT")

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
        # logging.basicConfig(level=logging.INFO)

    def provide_session(self) -> Session:
        """
        Provide a new database session.

        Returns:
            Session: A new SQLAlchemy session object.
        """
        base_session = sessionmaker(bind=self.engine)
        self.logger.debug("Providing new database session")
        return base_session()

    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope for database operations.

        Yields:
            Session: A new session within a transactional scope.
        """
        session = self.provide_session()
        self.logger.debug("Session scope opened")
        try:
            yield session
            session.commit()
            self.logger.debug("Transaction committed")
        except Exception as e:
            session.rollback()
            self.logger.error("Transaction failed", exc_info=True)
            raise
        finally:
            session.close()
            self.logger.debug("Session closed")


class AnalystTableManager(PostgresManager):
    """
    A specialized handler for analyst data in a PostgreSQL database.

    Inherits from PostgresManager to provide database interaction methods.
    """

    def __init__(self, db_host=None, db_name=None, db_user=None, db_password=None, db_port=None):
        """
        Initialize the CompanyDataHandler with database connection parameters.
        """
        super().__init__(db_host, db_name, db_user, db_password, db_port)
        self.logger.debug("CompanyTableManager ready.")

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name (str): The name of the table.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        self.logger.debug("Checking existence of table '%s'", table_name)
        inspector = inspect(self.engine)
        exists = table_name in inspector.get_table_names()
        self.logger.debug("Table '%s' exists: %s", table_name, exists)
        return exists

    def verify_table(self, table: Type, limit: int = 10, filters: Dict = None):
        """
        Verify the contents of a table by querying rows.

        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.
            limit (int): Maximum number of rows to fetch. Defaults to 10.
            filters (Dict): Optional filters as column-value pairs.
        """
        self.logger.debug(
            "Verifying table %s with limit=%d filters=%s",
            table.__tablename__,
            limit,
            filters,
        )
        with self.session_scope() as session:
            query = session.query(table)
            if filters:
                for column, value in filters.items():
                    query = query.filter(getattr(table, column) == value)
            rows = query.limit(limit).all()

            if not rows:
                self.logger.info("The table %s is empty.", table.__tablename__)
                return

            self.logger.info("Verified %d rows from %s", len(rows), table.__tablename__)
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
        self.logger.debug("Listing tables in the database")
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        self.logger.debug("Found tables: %s", tables)
        return tables

    def create_table(self, table: Type):
        """
        Create a table if it does not exist.

        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.
        """
        self.logger.debug("Creating table %s", table.__tablename__)
        table.__table__.create(bind=self.engine, checkfirst=True)
        self.logger.info("Table %s created", table.__tablename__)

    def drop_table(self, table: Type):
        """
        Drop a table if it exists.

        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.
        """
        self.logger.debug("Dropping table %s", table.__tablename__)
        table.__table__.drop(bind=self.engine, checkfirst=True)
        self.logger.info("Table %s dropped", table.__tablename__)


class AnalystDataManager(AnalystTableManager):
    """
    A specialized handler for analyst data in a PostgreSQL database.

    Inherits from PostgresManager to provide database interaction methods.
    """

    def __init__(self, db_host=None, db_name=None, db_user=None, db_password=None, db_port=None):
        """
        Initialize the CompanyDataHandler with database connection parameters.
        """
        super().__init__(db_host, db_name, db_user, db_password, db_port)
        self.logger.debug("CompanyDataManager ready.")

    def _row_to_dict(self, row):
        """
        Convert an SQLAlchemy ORM row to a dictionary.
        """
        return {
            column.key: getattr(row, column.key)
            for column in inspect(row).mapper.column_attrs
        }

    def get_existing_times(self, table: Type) -> set:
        """
        Fetch existing `time` values from the table.

        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.

        Returns:
            set: A set of existing `time` values in the table.
        """
        self.logger.debug("Fetching existing times from %s", table.__tablename__)
        with self.session_scope() as session:
            times = {row.time for row in session.query(table.time).all()}
        self.logger.debug(
            "Fetched %d existing times from %s", len(times), table.__tablename__
        )
        return times

    def get_time_range(self, table: Type) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get the minimum and maximum `time` values in the table.

        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.

        Returns:
            tuple: A tuple containing the minimum and maximum `time` values as pandas.Timestamp objects.
        """
        self.logger.debug("Fetching time range for %s", table.__tablename__)
        with self.session_scope() as session:
            min_time = session.query(func.min(table.time)).scalar()
            max_time = session.query(func.max(table.time)).scalar()
        self.logger.debug(
            "Time range for %s: %s - %s",
            table.__tablename__,
            min_time,
            max_time,
        )
        return (
            pd.Timestamp(min_time) if min_time else None,
            pd.Timestamp(max_time) if max_time else None,
        )

    def append_and_sort_data(self, table: Type, rows: List[Dict]):
        """
        Append new rows to the table if their primary key values do not already exist in the database.
        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.
            rows (List[Dict]): A list of dictionaries representing rows to append.
        """
        self.logger.debug(
            "Appending data for %s: %d incoming rows",
            table.__tablename__,
            len(rows),
        )
        if not rows:
            self.logger.info(
                f"No data provided for appending into {table.__tablename__}."
            )
            return

        # Determine primary key columns
        pk_cols = [col.name for col in table.__table__.primary_key.columns]
        if not pk_cols:
            raise ValueError(f"No primary key defined for table {table.__tablename__}")

        with self.session_scope() as session:
            # Build filter for existing PKs
            pk_tuples = [tuple(row[pk] for pk in pk_cols) for row in rows]
            # Build ORM filter for existing PKs
            if len(pk_cols) == 1:
                pk_col = getattr(table, pk_cols[0])
                existing = set(
                    r[0] for r in session.query(pk_col).filter(pk_col.in_([pk[0] for pk in pk_tuples])).all()
                )
                new_rows = [row for row in rows if row[pk_cols[0]] not in existing]
            else:
                # Multi-column PK: use sqlalchemy.tuple_
                existing = set(
                    tuple(getattr(r, pk) for pk in pk_cols)
                    for r in session.query(table).filter(
                        tuple_(*(getattr(table, pk) for pk in pk_cols)).in_(pk_tuples)
                    ).all()
                )
                new_rows = [row for row in rows if tuple(row[pk] for pk in pk_cols) not in existing]

            if not new_rows:
                self.logger.info(f"No new data to append into {table.__tablename__}.")
                return

            # Insert only new rows using ORM
            session.add_all([table(**row) for row in new_rows])
            self.logger.info(
                "Appended %d new rows into %s",
                len(new_rows),
                table.__tablename__,
            )
            # No physical sort is needed; always use ORDER BY in queries for sorted results.
            # ORM does not support physical reordering; this is handled at query time.
            self.logger.info(
                "Table %s is logically sorted by using ORDER BY in queries.",
                table.__tablename__,
            )

    def insert_new_data(self, table: Type, rows: List[Dict]):
        """
        Insert only new rows into the table based on primary key uniqueness, using ORM methods only.
        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.
            rows (List[Dict]): A list of dictionaries representing rows to insert.
        """
        self.logger.debug("Inserting %d rows into %s", len(rows), table.__tablename__)
        if not rows:
            self.logger.info(
                f"No data provided for insertion into {table.__tablename__}."
            )
            return

        pk_cols = [col.name for col in table.__table__.primary_key.columns]
        if not pk_cols:
            raise ValueError(f"No primary key defined for table {table.__tablename__}")

        with self.session_scope() as session:
            # Clean integer columns
            int_cols = [c.key for c in inspect(table).c if isinstance(c.type, (Integer, BigInteger))]
            for row in rows:
                for col in int_cols:
                    if col in row and (row[col] in ["-", "", "None", None] or pd.isna(row[col])):
                        row[col] = None

            # Build filter for existing PKs
            pk_tuples = [tuple(row[pk] for pk in pk_cols) for row in rows]
            if len(pk_cols) == 1:
                pk_col = getattr(table, pk_cols[0])
                existing = set(
                    r[0] for r in session.query(pk_col).filter(pk_col.in_([pk[0] for pk in pk_tuples])).all()
                )
                new_rows = [row for row in rows if row[pk_cols[0]] not in existing]
            else:
                # Multi-column PK: use sqlalchemy.tuple_
                existing = set(
                    tuple(getattr(r, pk) for pk in pk_cols)
                    for r in session.query(table).filter(
                        tuple_(*(getattr(table, pk) for pk in pk_cols)).in_(pk_tuples)
                    ).all()
                )
                new_rows = [row for row in rows if tuple(row[pk] for pk in pk_cols) not in existing]

            if not new_rows:
                self.logger.info(f"No new data to insert into {table.__tablename__}.")
                return

            session.add_all([table(**row) for row in new_rows])
            self.logger.info(
                "Inserted %d new rows into %s",
                len(new_rows),
                table.__tablename__,
            )

    def load_all(self, table: Type, limit: int = None) -> List[Dict]:
        """
        Fetch all rows from a table and return as a list of dictionaries.

        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.
            limit (int): Optional maximum number of rows to fetch.

        Returns:
            List[Dict]: A list of rows as dictionaries.
        """
        self.logger.debug(
            "Loading rows from %s with limit=%s", table.__tablename__, limit
        )
        with self.session_scope() as session:
            query = session.query(table)
            if limit:
                query = query.limit(limit)
            rows = query.all()

            self.logger.info("Loaded %d rows from %s", len(rows), table.__tablename__)

            # Convert ORM objects to dictionaries
            return [self._row_to_dict(row) for row in rows]

    def load_filtered_with_matching_values(
        self, table: Type, 
        filters: Dict
    ) -> List[Dict]:
        """
        Fetch rows from a table based on filter criteria.

        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.
            filters (Dict): Filters as column-value pairs.

        Returns:
            List[Dict]: A list of rows as dictionaries.
        """
        self.logger.debug(
            "Loading rows from %s with filters=%s",
            table.__tablename__,
            filters,
        )
        with self.session_scope() as session:
            query = session.query(table)
            for column, value in filters.items():
                query = query.filter(getattr(table, column) == value)
            rows = query.all()
            self.logger.info(
                "Loaded %d filtered rows from %s", len(rows), table.__tablename__
            )
            # Convert ORM objects to dictionaries
            return [self._row_to_dict(row) for row in rows]

