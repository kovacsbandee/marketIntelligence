

# Query the table
session = Session()
result = session.query(companyBase).all()
for row in result:
    print(f"company_ID: {row.company_ID}, Name: {row.company_name}, symbol: {row.symbol}, sector: {row.sector}, industry: {row.industry}")
session.close()
