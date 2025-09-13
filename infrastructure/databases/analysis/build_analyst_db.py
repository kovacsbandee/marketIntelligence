"""
Initializes all PostgreSQL tables defined in the analyst_table_objects module.

This script uses AnalystTableManager to:
    - Establish a connection to the analyst database (reading configuration from environment variables)
    - Discover all ORM classes defined in analyst_table_objects
    - Create corresponding tables in the PostgreSQL database
    - Log the names of all tables currently present in the database

Usage:
    Run this script directly to perform a one-time schema setup:
        $ python build_analyst_db.py
"""

import inspect
from infrastructure.databases.analysis.postgre_manager.analyst_data_manager import AnalystTableManager
from infrastructure.databases.analysis.postgre_manager import analyst_table_objects
from utils.logger import get_logger

def main(force_recreate=True):
    """
    Main entry point for analyst database schema creation.

    Initializes the database adapter, inspects table models in analyst_table_objects,
    creates the corresponding tables, and logs existing tables.
    """
    logger = get_logger("build_analyst_db")

    try:
        table_manager = AnalystTableManager()
    except ValueError as e:
        logger.error(f"Failed to initialize AnalystTableManager: {e}", exc_info=True)
        return

    logger.info("Successfully connected to the analyst PostgreSQL database.")

    # Introspect analyst_table_objects to get only locally defined ORM table classes
    classes = inspect.getmembers(analyst_table_objects, inspect.isclass)
    defined_classes = [
        cls[1] for cls in classes if cls[1].__module__ == analyst_table_objects.__name__
    ]

    logger.info("Found %d analyst table classes to create.", len(defined_classes))
    logger.debug("Analyst tables to create: %s", [cls.__name__ for cls in defined_classes])

    for table_class in defined_classes:
        logger.info("Creating table for: %s", table_class.__name__)
        if force_recreate:
            table_manager.drop_table(table=table_class)
        table_manager.create_table(table=table_class)

    logger.info("Created %d analyst tables", len(defined_classes))

    tables = table_manager.list_tables()
    logger.info("Tables currently in the analyst database: %s", tables)


if __name__ == "__main__":
    # Set force_recreate=True to drop and recreate all tables
    main(force_recreate=True)
