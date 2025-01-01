import yfinance as yf
from sqlalchemy import create_engine

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

print(f"Connecting to {DB_HOST} on port {DB_PORT}...")

connection_str = f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
# Create an engine for PostgreSQL
engine = create_engine(connection_str)


# Connect to the database
connection = engine.connect()

# Example: Create a table
connection.execute("""
    CREATE TABLE IF NOT EXISTS test_table (
        id SERIAL PRIMARY KEY,
        name VARCHAR(50),
        value INT
    )
""")

# Example: Insert data
connection.execute("INSERT INTO test_table (name, value) VALUES ('test_name', 123)")

# Example: Query data
result = connection.execute("SELECT * FROM test_table")
for row in result:
    print(row)

# Close the connection
connection.close()
