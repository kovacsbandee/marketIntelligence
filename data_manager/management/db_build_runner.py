import inspect
import logging
from data_manager.db_builders.postgre_adapter import PostgresAdapter
from data_manager.db_builders import postgre_objects

def main():
    # Setup basic logging (optional, improves readability)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DBInitializer")

    # Initialize Postgres Adapter (reads from env vars)
    try:
        adapter = PostgresAdapter()
    except ValueError as e:
        logger.error(f"Failed to initialize PostgresAdapter: {e}")
        return

    logger.info("Successfully connected to the PostgreSQL database.")

    # Inspect postgre_objects module to get all defined classes (table models)
    classes = inspect.getmembers(postgre_objects, inspect.isclass)

    # Filter out only those classes defined in postgre_objects (not imported ones)
    defined_classes = [
        cls[1] for cls in classes
        if cls[1].__module__ == postgre_objects.__name__
    ]

    logger.info(f"Found {len(defined_classes)} table classes to create.")

    # Iterate over each table class and create its table in the DB
    for table_class in defined_classes:
        logger.info(f"Creating table for: {table_class.__name__}")
        adapter.create_table(table=table_class)

    # List all tables that exist in the database after creation
    tables = adapter.list_tables()
    logger.info(f"Tables currently in the database: {tables}")

if __name__ == "__main__":
    main()