"""
Drops all PostgreSQL tables defined in the `postgre_objects` module, then recreates them.

Usage:
    $ python drop_and_build_db.py

WARNING:
    This operation is irreversible! All data in the specified tables will be lost.
"""

import inspect
from infrastructure.databases.company.postgre_manager.postgre_manager import CompanyTableManager
from infrastructure.databases.company.postgre_manager import postgre_objects
from utils.logger import get_logger

def drop_and_build():
    logger = get_logger("db_drop_and_build_runner")

    try:
        table_manager = CompanyTableManager()
    except ValueError as e:
        logger.error(f"Failed to initialize PostgresAdapter: {e}", exc_info=True)
        return

    logger.info("Connected to the database successfully.")

    # Discover ORM classes defined in postgre_objects
    classes = inspect.getmembers(postgre_objects, inspect.isclass)
    defined_classes = [
        cls[1] for cls in classes if cls[1].__module__ == postgre_objects.__name__
    ]

    if not defined_classes:
        logger.info("No table classes found to drop/build.")
        return

    # List tables to be dropped
    warning_msg = f"\n⚠️  WARNING: You are about to DELETE and RECREATE {len(defined_classes)} tables:"
    print(warning_msg)
    logger.warning(warning_msg)
    for table_class in defined_classes:
        print(f" - {table_class.__tablename__}")
    logger.debug("Tables to drop/build: %s", [cls.__tablename__ for cls in defined_classes])

    # Prompt for confirmation
    confirmation = input("\nType 'DELETE' to confirm and drop/recreate all tables: ")
    if confirmation != "DELETE":
        logger.info("Aborted. No tables were deleted or recreated.")
        return

    # Drop each table
    for table_class in defined_classes:
        logger.info("Dropping table: %s", table_class.__tablename__)
        table_manager.drop_table(table_class)

    logger.info("Dropped %d tables", len(defined_classes))

    # Recreate each table
    for table_class in defined_classes:
        logger.info("Creating table for: %s", table_class.__name__)
        table_manager.create_table(table=table_class)

    logger.info("Created %d tables", len(defined_classes))

    tables = table_manager.list_tables()
    logger.info("Tables currently in the database: %s", tables)
    print("\n✅ Drop and build completed. Current tables:", tables)

if __name__ == "__main__":
    drop_and_build()