import uuid
from typing import TypedDict
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class CompanyRow(TypedDict):
    company_ID: str
    symbol: str
    company_name: str
    market_cap: int
    country: str
    ipo_year: int
    sector: str
    industry: str


class Company(Base):
    __tablename__ = "company"
    company_ID = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String, nullable=False)
    company_name = Column(String, nullable=False)
    market_cap = Column(Integer, nullable=True)
    country = Column(String, nullable=True)
    ipo_year = Column(Integer, nullable=True)
    sector = Column(String, nullable=True)
    industry = Column(String, nullable=True)
