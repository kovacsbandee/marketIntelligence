import os
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from data_manager.build_db.db_objects import companyBase
# Load environment variables from .env file
load_dotenv()


# Access the environment variables
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")


# Define a sample table


db_connection_str = f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
# Create an engine for PostgreSQL
engine = create_engine(db_connection_str)

Base = declarative_base()
# Create the table (if it doesn't already exist)
Base.metadata.create_all(engine)

# Connect to the database
connection = engine.connect()

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Insert a test row
new_row = companyBase(company_name="TestStock", 
                      symbol="test",
                      sector="tech",
                      industry="IT")
session.add(new_row)

# Commit the transaction
try:
    session.commit()
    print("Test row inserted successfully!")
except Exception as e:
    session.rollback()  # Rollback in case of error
    print(f"Error inserting row: {e}")
finally:
    session.close()
