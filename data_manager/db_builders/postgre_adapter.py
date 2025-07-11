import os
import logging
from typing import List, Dict, Type
from contextlib import contextmanager

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, func, Integer, BigInteger
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import insert

# from data_manager.build_db.db_objects import DynamicCandlestickTable

# Load environment variables from .env file
load_dotenv()


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
        Append new rows to the table if their `time` values do not already exist in the database,
        and sort the table by `time`.

        Args:
            table (Type): The SQLAlchemy ORM-mapped class representing the table.
            rows (List[Dict]): A list of dictionaries representing rows to append.
        """
        self.logger.debug(
            "Appending and sorting data for %s: %d incoming rows",
            table.__tablename__,
            len(rows),
        )
        if not rows:
            self.logger.info(
                f"No data provided for appending into {table.__tablename__}."
            )
            return

        with self.session_scope() as session:
            # Step 1: Extract `time` values from the incoming rows
            incoming_times = {pd.Timestamp(row["time"]) for row in rows}

            # Step 2: Fetch only the existing `time` values from the database
            existing_times_query = session.query(table.time).filter(
                table.time.in_(incoming_times)
            )
            existing_times = {
                pd.Timestamp(time[0]) for time in existing_times_query.all()
            }

            # Step 3: Filter out rows with `time` values that already exist in the database
            new_rows = [
                row for row in rows if pd.Timestamp(row["time"]) not in existing_times
            ]

            if not new_rows:
                self.logger.info(f"No new data to append into {table.__tablename__}.")
                return

            # Step 4: Insert only the new rows
            session.bulk_insert_mappings(table, new_rows)
            self.logger.info(
                "Appended %d new rows into %s",
                len(new_rows),
                table.__tablename__,
            )

            # Step 5: Sort the table by `time` (order rows physically in the database)
            session.execute(
                f"""
                CREATE TABLE temp_sorted AS
                SELECT * FROM {table.__tablename__} ORDER BY time;
            """
            )
            session.execute(f"DROP TABLE {table.__tablename__};")
            session.execute(f"ALTER TABLE temp_sorted RENAME TO {table.__tablename__};")

            # Step 6: Reapply indexes or primary keys
            session.execute(
                f"ALTER TABLE {table.__tablename__} ADD PRIMARY KEY (time);"
            )
            self.logger.info(
                "Sorted table %s by `time` and reapplied indexes.",
                table.__tablename__,
            )

    def insert_new_data(self, table: Type, rows: List[Dict]):
        """
        Insert only new rows into the table based on time uniqueness.

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

        with self.session_scope() as session:
            try:
                session.bulk_insert_mappings(table, rows)
                self.logger.info(
                    "Inserted %d new rows into %s",
                    len(rows),
                    table.__tablename__,
                )
            except Exception as e:
                self.logger.error(
                    "Failed to insert new rows into %s",
                    table.__tablename__,
                    exc_info=True,
                )
                # Handle invalid integer placeholders like "-" by cleaning and retrying once
                if "invalid input syntax for type integer" in str(e):
                    self.logger.info("Retrying after sanitizing integer columns...")
                    int_cols = [
                        c.key
                        for c in inspect(table).c
                        if isinstance(c.type, (Integer, BigInteger))
                    ]
                    for row in rows:
                        for col in int_cols:
                            if col in row and row[col] == "-":
                                row[col] = None
                    try:
                        session.bulk_insert_mappings(table, rows)
                        self.logger.info(
                            "Inserted %d rows into %s after sanitizing",
                            len(rows),
                            table.__tablename__,
                        )
                    except Exception:
                        self.logger.error(
                            "Retry failed for inserting rows into %s",
                            table.__tablename__,
                            exc_info=True,
                        )
                        raise
                else:
                    raise

    # def insert_new_data(self, table: Type, rows: List[Dict]):
    #     """
    #     Insert rows into the table. If a row already exists (based on PK), do nothing.
    #     """
    #     if not rows:
    #         self.logger.info(f"No data provided for insertion into {table.__tablename__}.")
    #         return

    #     with self.session_scope() as session:
    #         try:
    #             stmt = insert(table).values(rows)
    #             stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "fiscal_date_ending"])
    #             session.execute(stmt)
    #             self.logger.info(f"Inserted rows into {table.__tablename__} (skipping duplicates).")
    #         except Exception as e:
    #             self.logger.error(f"Failed to insert rows into {table.__tablename__}: {e}")
    #             raise

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
        self, table: Type, filters: Dict
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

    def _row_to_dict(self, row):
        """
        Convert an SQLAlchemy ORM row to a dictionary.
        """
        return {
            column.key: getattr(row, column.key)
            for column in inspect(row).mapper.column_attrs
        }
