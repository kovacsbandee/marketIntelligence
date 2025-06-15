"""
Drops all PostgreSQL tables defined in the `postgre_objects` module.

This script uses the PostgresAdapter to:
    - Connect to the PostgreSQL database (using environment variables)
    - Discover all SQLAlchemy ORM table classes defined in `data_manager.db_builders.postgre_objects`
    - Prompt the user for confirmation before dropping any tables
    - Drop each of the discovered tables from the database
    - Print and log the outcome of the operation

Usage:
    Run this script directly to perform a destructive schema teardown:
        $ python db_drop_runner.py

Dependencies:
    - The PostgresAdapter class must be properly implemented and accessible.
    - Environment variables (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT) must be set.
    - SQLAlchemy ORM classes should be defined in `postgre_objects`.

WARNING:
    This operation is irreversible! All data in the specified tables will be lost.
"""

import inspect
import logging
from data_manager.db_builders.postgre_adapter import PostgresAdapter
from data_manager.db_builders import postgre_objects


def main():
    """
    Main entry point for database schema teardown.

    Connects to the database, discovers table models in `postgre_objects`,
    prompts for user confirmation, and drops the corresponding tables.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DBDropper")

    try:
        adapter = PostgresAdapter()
    except ValueError as e:
        logger.error("Failed to initialize PostgresAdapter: %s", e)
        return

    logger.info("Connected to the database successfully.")

    # Discover ORM classes defined in postgre_objects
    classes = inspect.getmembers(postgre_objects, inspect.isclass)
    defined_classes = [
        cls[1] for cls in classes if cls[1].__module__ == postgre_objects.__name__
    ]

    if not defined_classes:
        logger.info("No table classes found to drop.")
        return

    # List tables to be dropped
    print(f"\n⚠️  WARNING: You are about to delete {len(defined_classes)} tables:")
    for table_class in defined_classes:
        print(f" - {table_class.__tablename__}")

    # Prompt for confirmation
    confirmation = input("\nType 'DELETE' to confirm and drop all tables: ")
    if confirmation != "DELETE":
        print("Aborted. No tables were deleted.")
        return

    # Drop each table and log progress
    for table_class in defined_classes:
        print(f"Dropping table: {table_class.__tablename__}")
        logger.info("Dropping table: %s", table_class.__tablename__)
        adapter.drop_table(table_class)

    print("✅ All specified tables have been dropped.")
    logger.info("All specified tables have been dropped.")


if __name__ == "__main__":
    main()
