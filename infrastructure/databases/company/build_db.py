"""
Initializes all PostgreSQL tables defined in the `postgre_objects` module.

This script uses the PostgresAdapter to:
    - Establish a connection to the database (reading configuration from environment variables)
    - Discover all ORM classes defined in `data_manager.db_builders.postgre_objects`
    - Create corresponding tables in the PostgreSQL database
    - Log the names of all tables currently present in the database

Usage:
    Run this script directly to perform a one-time schema setup:
        $ python db_build_runner.py

Dependencies:
    - The PostgresAdapter class must be properly implemented and accessible.
    - Environment variables (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT) must be set.
    - SQLAlchemy ORM classes should be defined in `postgre_objects`.
"""

import inspect
from infrastructure.databases.company.postgre_manager.company_data_manager import CompanyTableManager
from infrastructure.databases.company.postgre_manager import company_table_objects
from utils.logger import get_logger


def main(force_recreate=True):
    """
    Main entry point for database schema creation.

    Initializes the database adapter, inspects table models in `postgre_objects`,
    creates the corresponding tables, and logs existing tables.
    """
    logger = get_logger("db_build_runner")

    try:
        table_manager = CompanyTableManager()
    except ValueError as e:
        logger.error(f"Failed to initialize PostgresAdapter: {e}", exc_info=True)
        return

    logger.info("Successfully connected to the PostgreSQL database.")

    # Introspect postgre_objects to get only locally defined ORM table classes
    classes = inspect.getmembers(company_table_objects, inspect.isclass)
    defined_classes = [
        cls[1] for cls in classes if cls[1].__module__ == company_table_objects.__name__
    ]

    logger.info("Found %d table classes to create.", len(defined_classes))
    logger.debug("Tables to create: %s", [cls.__name__ for cls in defined_classes])

    for table_class in defined_classes:
        logger.info("Creating table for: %s", table_class.__name__)
        if force_recreate:
            table_manager.drop_table(table=table_class)
        table_manager.create_table(table=table_class)

    logger.info("Created %d tables", len(defined_classes))

    tables = table_manager.list_tables()
    logger.info("Tables currently in the database: %s", tables)


if __name__ == "__main__":
    # Set force_recreate=True to drop and recreate all tables
    main(force_recreate=True)
