import inspect
import logging
from data_manager.build_db.postgre_adapter import PostgresAdapter
from data_manager.build_db import postgre_objects

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DBDropper")

    # Initialize the adapter
    try:
        adapter = PostgresAdapter()
    except ValueError as e:
        logger.error(f"Failed to initialize PostgresAdapter: {e}")
        return

    logger.info("Connected to the database successfully.")

    # Inspect and gather all ORM-mapped classes from postgre_objects
    classes = inspect.getmembers(postgre_objects, inspect.isclass)

    # Filter: only classes defined in postgre_objects, not imported ones
    defined_classes = [
        cls[1] for cls in classes
        if cls[1].__module__ == postgre_objects.__name__
    ]

    if not defined_classes:
        logger.info("No table classes found to drop.")
        return

    # Display list of tables to drop (table class names)
    print(f"\n⚠️  WARNING: You are about to delete {len(defined_classes)} tables:")
    for table_class in defined_classes:
        print(f" - {table_class.__tablename__}")

    # Prompt for confirmation
    confirmation = input("\nType 'DELETE' to confirm and drop all tables: ")

    if confirmation != "DELETE":
        print("Aborted. No tables were deleted.")
        return

    # Drop each table
    for table_class in defined_classes:
        print(f"Dropping table: {table_class.__tablename__}")
        adapter.drop_table(table_class)

    print("✅ All specified tables have been dropped.")

if __name__ == "__main__":
    main()
