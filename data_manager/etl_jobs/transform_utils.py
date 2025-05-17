import pandas as pd
import numpy as np

def standardize_company_fundamentals_columns(df):
    """Rename columns from Alpha Vantage to match the database schema."""

    column_map = {
    'Symbol': 'symbol',
    'AssetType': 'asset_type',
    'Name': 'name',
    'Description': 'description',
    'CIK': 'cik',
    'Exchange': 'exchange',
    'Currency': 'currency',
    'Country': 'country',
    'Sector': 'sector',
    'Industry': 'industry',
    'Address': 'address',
    'OfficialSite': 'official_site',
    'FiscalYearEnd': 'fiscal_year_end',
    'LatestQuarter': 'latest_quarter',
    'MarketCapitalization': 'market_capitalization',
    'EBITDA': 'ebitda',
    'PERatio': 'pe_ratio',
    'PEGRatio': 'peg_ratio',
    'BookValue': 'book_value',
    'DividendPerShare': 'dividend_per_share',
    'DividendYield': 'dividend_yield',
    'EPS': 'eps',
    'RevenuePerShareTTM': 'revenue_per_share_ttm',
    'ProfitMargin': 'profit_margin',
    'OperatingMarginTTM': 'operating_margin_ttm',
    'ReturnOnAssetsTTM': 'return_on_assets_ttm',
    'ReturnOnEquityTTM': 'return_on_equity_ttm',
    'RevenueTTM': 'revenue_ttm',
    'GrossProfitTTM': 'gross_profit_ttm',
    'DilutedEPSTTM': 'diluted_eps_ttm',
    'QuarterlyEarningsGrowthYOY': 'quarterly_earnings_growth_yoy',
    'QuarterlyRevenueGrowthYOY': 'quarterly_revenue_growth_yoy',
    'AnalystTargetPrice': 'analyst_target_price',
    'AnalystRatingStrongBuy': 'analyst_rating_strong_buy',
    'AnalystRatingBuy': 'analyst_rating_buy',
    'AnalystRatingHold': 'analyst_rating_hold',
    'AnalystRatingSell': 'analyst_rating_sell',
    'AnalystRatingStrongSell': 'analyst_rating_strong_sell',
    'TrailingPE': 'trailing_pe',
    'ForwardPE': 'forward_pe',
    'PriceToSalesRatioTTM': 'price_to_sales_ratio_ttm',
    'PriceToBookRatio': 'price_to_book_ratio',
    'EVToRevenue': 'ev_to_revenue',
    'EVToEBITDA': 'ev_to_ebitda',
    'Beta': 'beta',
    '52WeekHigh': 'fifty_two_week_high',
    '52WeekLow': 'fifty_two_week_low',
    '50DayMovingAverage': 'fifty_day_moving_average',
    '200DayMovingAverage': 'two_hundred_day_moving_average',
    'SharesOutstanding': 'shares_outstanding',
    'DividendDate': 'dividend_date',
    'ExDividendDate': 'ex_dividend_date'
}

    return df.rename(columns=column_map)


INCOME_STATEMENT_NUMERIC_COLUMNS = [
    'gross_profit', 'total_revenue', 'cost_of_revenue', 'cost_of_goods_and_services_sold',
    'operating_income', 'selling_general_and_administrative', 'research_and_development',
    'operating_expenses', 'investment_income_net', 'net_interest_income', 'interest_income',
    'interest_expense', 'non_interest_income', 'other_non_operating_income', 'depreciation',
    'depreciation_and_amortization', 'income_before_tax', 'income_tax_expense',
    'interest_and_debt_expense', 'net_income_from_continuing_operations',
    'comprehensive_income_net_of_tax', 'ebit', 'ebitda', 'net_income'
]

def standardize_annual_income_statement_columns(df):
    """Rename and clean Alpha Vantage annual income statement columns for DB insert."""
    column_map = {
        'symbol': 'symbol',
        'fiscalDateEnding': 'fiscal_date_ending',
        'reportedCurrency': 'reported_currency',
        'grossProfit': 'gross_profit',
        'totalRevenue': 'total_revenue',
        'costOfRevenue': 'cost_of_revenue',
        'costofGoodsAndServicesSold': 'cost_of_goods_and_services_sold',
        'operatingIncome': 'operating_income',
        'sellingGeneralAndAdministrative': 'selling_general_and_administrative',
        'researchAndDevelopment': 'research_and_development',
        'operatingExpenses': 'operating_expenses',
        'investmentIncomeNet': 'investment_income_net',
        'netInterestIncome': 'net_interest_income',
        'interestIncome': 'interest_income',
        'interestExpense': 'interest_expense',
        'nonInterestIncome': 'non_interest_income',
        'otherNonOperatingIncome': 'other_non_operating_income',
        'depreciation': 'depreciation',
        'depreciationAndAmortization': 'depreciation_and_amortization',
        'incomeBeforeTax': 'income_before_tax',
        'incomeTaxExpense': 'income_tax_expense',
        'interestAndDebtExpense': 'interest_and_debt_expense',
        'netIncomeFromContinuingOperations': 'net_income_from_continuing_operations',
        'comprehensiveIncomeNetOfTax': 'comprehensive_income_net_of_tax',
        'ebit': 'ebit',
        'ebitda': 'ebitda',
        'netIncome': 'net_income',
    }
    df = df.rename(columns=column_map)
    # Clean and convert values
    df.replace(
        to_replace=["None", "none", "NaN", "nan", ""],
        value=np.nan,
        inplace=True
    )
    df = df.infer_objects(copy=False)  # Explicitly call infer_objects to avoid future warning

    for col in INCOME_STATEMENT_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors="coerce")
    df = df.where(pd.notnull(df), None)
    return df

def standardize_quarterly_income_statement_columns(df):
    """Rename and clean Alpha Vantage quarterly income statement columns for DB insert."""
    column_map = {
        'symbol': 'symbol',
        'fiscalDateEnding': 'fiscal_date_ending',
        'reportedCurrency': 'reported_currency',
        'grossProfit': 'gross_profit',
        'totalRevenue': 'total_revenue',
        'costOfRevenue': 'cost_of_revenue',
        'costofGoodsAndServicesSold': 'cost_of_goods_and_services_sold',
        'operatingIncome': 'operating_income',
        'sellingGeneralAndAdministrative': 'selling_general_and_administrative',
        'researchAndDevelopment': 'research_and_development',
        'operatingExpenses': 'operating_expenses',
        'investmentIncomeNet': 'investment_income_net',
        'netInterestIncome': 'net_interest_income',
        'interestIncome': 'interest_income',
        'interestExpense': 'interest_expense',
        'nonInterestIncome': 'non_interest_income',
        'otherNonOperatingIncome': 'other_non_operating_income',
        'depreciation': 'depreciation',
        'depreciationAndAmortization': 'depreciation_and_amortization',
        'incomeBeforeTax': 'income_before_tax',
        'incomeTaxExpense': 'income_tax_expense',
        'interestAndDebtExpense': 'interest_and_debt_expense',
        'netIncomeFromContinuingOperations': 'net_income_from_continuing_operations',
        'comprehensiveIncomeNetOfTax': 'comprehensive_income_net_of_tax',
        'ebit': 'ebit',
        'ebitda': 'ebitda',
        'netIncome': 'net_income',
    }
    df = df.rename(columns=column_map)
    df.replace(
        to_replace=["None", "none", "NaN", "nan", ""],
        value=np.nan,
        inplace=True
    )
    df = df.infer_objects(copy=False)  # Avoids future warning

    for col in INCOME_STATEMENT_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors="coerce")
    df = df.where(pd.notnull(df), None)
    return df


def standardize_annual_balance_sheet_columns(df):
    """Rename Alpha Vantage annual balance sheet columns to match database schema."""
    column_map = {
        'symbol': 'symbol',
        'fiscalDateEnding': 'fiscal_date_ending',
        'reportedCurrency': 'reported_currency',
        'totalAssets': 'total_assets',
        'totalCurrentAssets': 'total_current_assets',
        'cashAndCashEquivalentsAtCarryingValue': 'cash_and_cash_equivalents',
        'cashAndShortTermInvestments': 'cash_and_short_term_investments',
        'inventory': 'inventory',
        'currentNetReceivables': 'current_net_receivables',
        'totalNonCurrentAssets': 'total_non_current_assets',
        'propertyPlantEquipment': 'property_plant_equipment',
        'accumulatedDepreciationAmortizationPPE': 'accumulated_depreciation_amortization_ppe',
        'intangibleAssets': 'intangible_assets',
        'goodwill': 'goodwill',
        'longTermInvestments': 'long_term_investments',
        'shortTermInvestments': 'short_term_investments',
        'otherCurrentAssets': 'other_current_assets',
        'otherNonCurrentAssets': 'other_non_current_assets',
        'totalLiabilities': 'total_liabilities',
        'totalCurrentLiabilities': 'total_current_liabilities',
        'currentAccountsPayable': 'current_accounts_payable',
        'deferredRevenue': 'deferred_revenue',
        'currentDebt': 'current_debt',
        'shortTermDebt': 'short_term_debt',
        'totalNonCurrentLiabilities': 'total_non_current_liabilities',
        'capitalLeaseObligations': 'capital_lease_obligations',
        'longTermDebt': 'long_term_debt',
        'shortLongTermDebtTotal': 'short_long_term_debt_total',
        'otherCurrentLiabilities': 'other_current_liabilities',
        'otherNonCurrentLiabilities': 'other_non_current_liabilities',
        'totalShareholderEquity': 'total_shareholder_equity',
        'treasuryStock': 'treasury_stock',
        'retainedEarnings': 'retained_earnings',
        'commonStock': 'common_stock',
        'commonStockSharesOutstanding': 'common_stock_shares_outstanding',
    }
    return df.rename(columns=column_map)

def standardize_quarterly_balance_sheet_columns(df):
    """Rename Alpha Vantage quarterly balance sheet columns to match database schema."""
    column_map = {
        'symbol': 'symbol',
        'fiscalDateEnding': 'fiscal_date_ending',
        'reportedCurrency': 'reported_currency',
        'totalAssets': 'total_assets',
        'totalCurrentAssets': 'total_current_assets',
        'cashAndCashEquivalentsAtCarryingValue': 'cash_and_cash_equivalents',
        'cashAndShortTermInvestments': 'cash_and_short_term_investments',
        'inventory': 'inventory',
        'currentNetReceivables': 'current_net_receivables',
        'totalNonCurrentAssets': 'total_non_current_assets',
        'propertyPlantEquipment': 'property_plant_equipment',
        'accumulatedDepreciationAmortizationPPE': 'accumulated_depreciation_amortization_ppe',
        'intangibleAssets': 'intangible_assets',
        'goodwill': 'goodwill',
        'longTermInvestments': 'long_term_investments',
        'shortTermInvestments': 'short_term_investments',
        'otherCurrentAssets': 'other_current_assets',
        'otherNonCurrentAssets': 'other_non_current_assets',
        'totalLiabilities': 'total_liabilities',
        'totalCurrentLiabilities': 'total_current_liabilities',
        'currentAccountsPayable': 'current_accounts_payable',
        'deferredRevenue': 'deferred_revenue',
        'currentDebt': 'current_debt',
        'shortTermDebt': 'short_term_debt',
        'totalNonCurrentLiabilities': 'total_non_current_liabilities',
        'capitalLeaseObligations': 'capital_lease_obligations',
        'longTermDebt': 'long_term_debt',
        'shortLongTermDebtTotal': 'short_long_term_debt_total',
        'otherCurrentLiabilities': 'other_current_liabilities',
        'otherNonCurrentLiabilities': 'other_non_current_liabilities',
        'totalShareholderEquity': 'total_shareholder_equity',
        'treasuryStock': 'treasury_stock',
        'retainedEarnings': 'retained_earnings',
        'commonStock': 'common_stock',
        'commonStockSharesOutstanding': 'common_stock_shares_outstanding',
    }
    return df.rename(columns=column_map)


def standardize_annual_balance_sheet_columns(df):
    """Rename and type-cast columns from Alpha Vantage annual balance sheet to match the database schema."""

    column_map = {
        'symbol': 'symbol',
        'fiscalDateEnding': 'fiscal_date_ending',
        'reportedCurrency': 'reported_currency',
        'totalAssets': 'total_assets',
        'totalCurrentAssets': 'total_current_assets',
        'cashAndCashEquivalentsAtCarryingValue': 'cash_and_cash_equivalents',
        'cashAndShortTermInvestments': 'cash_and_short_term_investments',
        'inventory': 'inventory',
        'currentNetReceivables': 'current_net_receivables',
        'totalNonCurrentAssets': 'total_non_current_assets',
        'propertyPlantEquipment': 'property_plant_equipment',
        'accumulatedDepreciationAmortizationPPE': 'accumulated_depreciation_amortization_ppe',
        'intangibleAssets': 'intangible_assets',
        'goodwill': 'goodwill',
        'longTermInvestments': 'long_term_investments',
        'shortTermInvestments': 'short_term_investments',
        'otherCurrentAssets': 'other_current_assets',
        'otherNonCurrentAssets': 'other_non_current_assets',
        'totalLiabilities': 'total_liabilities',
        'totalCurrentLiabilities': 'total_current_liabilities',
        'currentAccountsPayable': 'current_accounts_payable',
        'deferredRevenue': 'deferred_revenue',
        'currentDebt': 'current_debt',
        'shortTermDebt': 'short_term_debt',
        'totalNonCurrentLiabilities': 'total_non_current_liabilities',
        'capitalLeaseObligations': 'capital_lease_obligations',
        'longTermDebt': 'long_term_debt',
        'shortLongTermDebtTotal': 'short_long_term_debt_total',
        'otherCurrentLiabilities': 'other_current_liabilities',
        'otherNonCurrentLiabilities': 'other_non_current_liabilities',
        'totalShareholderEquity': 'total_shareholder_equity',
        'treasuryStock': 'treasury_stock',
        'retainedEarnings': 'retained_earnings',
        'commonStock': 'common_stock',
        'commonStockSharesOutstanding': 'common_stock_shares_outstanding'
    }

    # Rename columns
    df = df.rename(columns=column_map)

    # Convert columns to appropriate types
    # Date
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')

    # Numeric columns (excluding 'symbol', 'reported_currency')
    float_columns = [
        'total_assets', 'total_current_assets', 'cash_and_cash_equivalents',
        'cash_and_short_term_investments', 'inventory', 'current_net_receivables',
        'total_non_current_assets', 'property_plant_equipment', 'accumulated_depreciation_amortization_ppe',
        'intangible_assets', 'goodwill', 'long_term_investments', 'short_term_investments',
        'other_current_assets', 'other_non_current_assets', 'total_liabilities',
        'total_current_liabilities', 'current_accounts_payable', 'deferred_revenue',
        'current_debt', 'short_term_debt', 'total_non_current_liabilities', 'capital_lease_obligations',
        'long_term_debt', 'short_long_term_debt_total', 'other_current_liabilities',
        'other_non_current_liabilities', 'total_shareholder_equity', 'treasury_stock',
        'retained_earnings', 'common_stock'
    ]
    int_columns = ['common_stock_shares_outstanding']

    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # Clean any 'None', 'nan', 'NaN', etc.
    df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan, inplace=True)

    return df

def standardize_quarterly_balance_sheet_columns(df):
    """Rename and type-cast columns from Alpha Vantage quarterly balance sheet to match the database schema."""

    column_map = {
        'symbol': 'symbol',
        'fiscalDateEnding': 'fiscal_date_ending',
        'reportedCurrency': 'reported_currency',
        'totalAssets': 'total_assets',
        'totalCurrentAssets': 'total_current_assets',
        'cashAndCashEquivalentsAtCarryingValue': 'cash_and_cash_equivalents',
        'cashAndShortTermInvestments': 'cash_and_short_term_investments',
        'inventory': 'inventory',
        'currentNetReceivables': 'current_net_receivables',
        'totalNonCurrentAssets': 'total_non_current_assets',
        'propertyPlantEquipment': 'property_plant_equipment',
        'accumulatedDepreciationAmortizationPPE': 'accumulated_depreciation_amortization_ppe',
        'intangibleAssets': 'intangible_assets',
        'intangibleAssetsExcludingGoodwill': 'intangible_assets_excluding_goodwill',
        'goodwill': 'goodwill',
        'investments': 'investments',
        'longTermInvestments': 'long_term_investments',
        'shortTermInvestments': 'short_term_investments',
        'otherCurrentAssets': 'other_current_assets',
        'otherNonCurrentAssets': 'other_non_current_assets',
        'totalLiabilities': 'total_liabilities',
        'totalCurrentLiabilities': 'total_current_liabilities',
        'currentAccountsPayable': 'current_accounts_payable',
        'deferredRevenue': 'deferred_revenue',
        'currentDebt': 'current_debt',
        'shortTermDebt': 'short_term_debt',
        'totalNonCurrentLiabilities': 'total_non_current_liabilities',
        'capitalLeaseObligations': 'capital_lease_obligations',
        'longTermDebt': 'long_term_debt',
        'currentLongTermDebt': 'current_long_term_debt',
        'longTermDebtNoncurrent': 'long_term_debt_noncurrent',
        'shortLongTermDebtTotal': 'short_long_term_debt_total',
        'otherCurrentLiabilities': 'other_current_liabilities',
        'otherNonCurrentLiabilities': 'other_non_current_liabilities',
        'totalShareholderEquity': 'total_shareholder_equity',
        'treasuryStock': 'treasury_stock',
        'retainedEarnings': 'retained_earnings',
        'commonStock': 'common_stock',
        'commonStockSharesOutstanding': 'common_stock_shares_outstanding'
    }

    df = df.rename(columns=column_map)

    # Date
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')

    float_columns = [
        'total_assets', 'total_current_assets', 'cash_and_cash_equivalents',
        'cash_and_short_term_investments', 'inventory', 'current_net_receivables',
        'total_non_current_assets', 'property_plant_equipment', 'accumulated_depreciation_amortization_ppe',
        'intangible_assets', 'intangible_assets_excluding_goodwill', 'goodwill',
        'investments', 'long_term_investments', 'short_term_investments', 'other_current_assets',
        'other_non_current_assets', 'total_liabilities', 'total_current_liabilities',
        'current_accounts_payable', 'deferred_revenue', 'current_debt', 'short_term_debt',
        'total_non_current_liabilities', 'capital_lease_obligations', 'long_term_debt',
        'current_long_term_debt', 'long_term_debt_noncurrent', 'short_long_term_debt_total',
        'other_current_liabilities', 'other_non_current_liabilities', 'total_shareholder_equity',
        'treasury_stock', 'retained_earnings', 'common_stock'
    ]
    int_columns = ['common_stock_shares_outstanding']

    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan, inplace=True)

    return df

import numpy as np
import pandas as pd

def standardize_annual_cash_flow_columns(df):
    """Rename and type-cast Alpha Vantage annual cash flow columns to DB schema."""
    column_map = {
        'symbol': 'symbol',
        'fiscalDateEnding': 'fiscal_date_ending',
        'reportedCurrency': 'reported_currency',
        'operatingCashflow': 'operating_cashflow',
        'paymentsForOperatingActivities': 'payments_for_operating_activities',
        'proceedsFromOperatingActivities': 'proceeds_from_operating_activities',
        'changeInOperatingLiabilities': 'change_in_operating_liabilities',
        'changeInOperatingAssets': 'change_in_operating_assets',
        'depreciationDepletionAndAmortization': 'depreciation_depletion_and_amortization',
        'capitalExpenditures': 'capital_expenditures',
        'changeInReceivables': 'change_in_receivables',
        'changeInInventory': 'change_in_inventory',
        'profitLoss': 'profit_loss',
        'cashflowFromInvestment': 'cashflow_from_investment',
        'cashflowFromFinancing': 'cashflow_from_financing',
        'proceedsFromRepaymentsOfShortTermDebt': 'proceeds_from_repayments_of_short_term_debt',
        'paymentsForRepurchaseOfCommonStock': 'payments_for_repurchase_of_common_stock',
        'paymentsForRepurchaseOfEquity': 'payments_for_repurchase_of_equity',
        'paymentsForRepurchaseOfPreferredStock': 'payments_for_repurchase_of_preferred_stock',
        'dividendPayout': 'dividend_payout',
        'dividendPayoutCommonStock': 'dividend_payout_common_stock',
        'dividendPayoutPreferredStock': 'dividend_payout_preferred_stock',
        'proceedsFromIssuanceOfCommonStock': 'proceeds_from_issuance_of_common_stock',
        'proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet': 'proceeds_from_issuance_of_long_term_debt_and_capital_securities_net',
        'proceedsFromIssuanceOfPreferredStock': 'proceeds_from_issuance_of_preferred_stock',
        'proceedsFromRepurchaseOfEquity': 'proceeds_from_repurchase_of_equity',
        'proceedsFromSaleOfTreasuryStock': 'proceeds_from_sale_of_treasury_stock',
        'changeInCashAndCashEquivalents': 'change_in_cash_and_cash_equivalents',
        'changeInExchangeRate': 'change_in_exchange_rate',
        'netIncome': 'net_income',
    }
    # Rename columns
    df = df.rename(columns=column_map)
    # Replace "None", "none", "", "NaN", "nan" with np.nan
    df = df.replace(
        to_replace=["None", "none", "NaN", "nan", ""],
        value=np.nan
    )
    df = df.infer_objects(copy=False)

    # Date conversion
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')

    # Float columns (all except symbol, date, currency)
    float_cols = [col for col in df.columns if col not in ['symbol', 'fiscal_date_ending', 'reported_currency']]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Clean up
    return df

def standardize_quarterly_cash_flow_columns(df):
    """Rename and type-cast Alpha Vantage quarterly cash flow columns to DB schema."""
    column_map = {
        'symbol': 'symbol',
        'fiscalDateEnding': 'fiscal_date_ending',
        'reportedCurrency': 'reported_currency',
        'operatingCashflow': 'operating_cashflow',
        'paymentsForOperatingActivities': 'payments_for_operating_activities',
        'proceedsFromOperatingActivities': 'proceeds_from_operating_activities',
        'changeInOperatingLiabilities': 'change_in_operating_liabilities',
        'changeInOperatingAssets': 'change_in_operating_assets',
        'depreciationDepletionAndAmortization': 'depreciation_depletion_and_amortization',
        'capitalExpenditures': 'capital_expenditures',
        'changeInReceivables': 'change_in_receivables',
        'changeInInventory': 'change_in_inventory',
        'profitLoss': 'profit_loss',
        'cashflowFromInvestment': 'cashflow_from_investment',
        'cashflowFromFinancing': 'cashflow_from_financing',
        'proceedsFromRepaymentsOfShortTermDebt': 'proceeds_from_repayments_of_short_term_debt',
        'paymentsForRepurchaseOfCommonStock': 'payments_for_repurchase_of_common_stock',
        'paymentsForRepurchaseOfEquity': 'payments_for_repurchase_of_equity',
        'paymentsForRepurchaseOfPreferredStock': 'payments_for_repurchase_of_preferred_stock',
        'dividendPayout': 'dividend_payout',
        'dividendPayoutCommonStock': 'dividend_payout_common_stock',
        'dividendPayoutPreferredStock': 'dividend_payout_preferred_stock',
        'proceedsFromIssuanceOfCommonStock': 'proceeds_from_issuance_of_common_stock',
        'proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet': 'proceeds_from_issuance_of_long_term_debt_and_capital_securities_net',
        'proceedsFromIssuanceOfPreferredStock': 'proceeds_from_issuance_of_preferred_stock',
        'proceedsFromRepurchaseOfEquity': 'proceeds_from_repurchase_of_equity',
        'proceedsFromSaleOfTreasuryStock': 'proceeds_from_sale_of_treasury_stock',
        'changeInCashAndCashEquivalents': 'change_in_cash_and_cash_equivalents',
        'changeInExchangeRate': 'change_in_exchange_rate',
        'netIncome': 'net_income',
    }
    # Rename columns
    df = df.rename(columns=column_map)
    # Replace "None", "none", "", "NaN", "nan" with np.nan
    df = df.replace(
        to_replace=["None", "none", "NaN", "nan", ""],
        value=np.nan
    )
    df = df.infer_objects(copy=False)

    # Date conversion
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')

    # Float columns (all except symbol, date, currency)
    float_cols = [col for col in df.columns if col not in ['symbol', 'fiscal_date_ending', 'reported_currency']]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Clean up
    return df

def standardize_annual_earnings_columns(df):
    """Rename Alpha Vantage annual earnings columns to match the database schema and convert types."""
    column_map = {
        'symbol': 'symbol',
        'fiscalDateEnding': 'fiscal_date_ending',
        'reportedEPS': 'reported_eps',
    }
    df = df.rename(columns=column_map)
    # Convert types
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')
    if 'reported_eps' in df.columns:
        df['reported_eps'] = pd.to_numeric(df['reported_eps'], errors='coerce')
    if 'symbol' in df.columns and 'symbol' in column_map:
        df['symbol'] = df['symbol'].astype(str)
    # Replace string nulls and convert
    df.replace(
        to_replace=["None", "none", "NaN", "nan", ""],
        value=np.nan,
        inplace=True
    )
    df = df.infer_objects(copy=False)
    return df

def standardize_quarterly_earnings_columns(df):
    """Rename Alpha Vantage quarterly earnings columns to match the database schema and convert types."""
    column_map = {
        'symbol': 'symbol',
        'fiscalDateEnding': 'fiscal_date_ending',
        'reportedDate': 'reported_date',
        'reportedEPS': 'reported_eps',
        'estimatedEPS': 'estimated_eps',
        'surprise': 'surprise',
        'surprisePercentage': 'surprise_percentage',
        'reportTime': 'report_time',
    }
    df = df.rename(columns=column_map)
    # Convert types
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')
    if 'reported_date' in df.columns:
        df['reported_date'] = pd.to_datetime(df['reported_date'], errors='coerce')
    float_cols = ['reported_eps', 'estimated_eps', 'surprise', 'surprise_percentage']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'report_time' in df.columns:
        df['report_time'] = df['report_time'].astype(str)
    if 'symbol' in df.columns and 'symbol' in column_map:
        df['symbol'] = df['symbol'].astype(str)
    df.replace(
        to_replace=["None", "none", "NaN", "nan", ""],
        value=np.nan,
        inplace=True
    )
    df = df.infer_objects(copy=False)
    return df


def standardize_insider_transaction_columns(df):
    """
    Standardizes column names and types for Alpha Vantage insider transaction data.
    """
    column_map = {
        "transaction_date": "transaction_date",
        "ticker": "symbol",
        "executive": "executive",
        "executive_title": "executive_title",
        "security_type": "security_type",
        "acquisition_or_disposal": "acquisition_or_disposal",
        "shares": "shares",
        "share_price": "share_price",
    }
    df = df.rename(columns=column_map)

    # Protect against missing columns
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    for col in ["shares", "share_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardize blanks/None/nulls
    df = df.replace(
        to_replace=["None", "none", "NaN", "nan", ""],
        value=np.nan
    )
    df = df.infer_objects(copy=False)
    df = df.where(pd.notnull(df), None)

    # Warn if required columns are missing
    required_cols = ["transaction_date", "symbol"]
    for c in required_cols:
        if c not in df.columns:
            print(f"❌ Warning: Required column '{c}' is missing after standardization!")
    df.drop_duplicates(
        subset=[
            "transaction_date",
            "symbol",
            "executive",
            "security_type",
            "acquisition_or_disposal"
        ],
        inplace=True)

    return df

def standardize_stock_split_columns(df):
    """
    Standardizes column names and types for stock split data.
    """
    column_map = {
        "effective_date": "effective_date",
        "split_factor": "split_factor",
        "symbol": "symbol",
    }
    df = df.rename(columns=column_map)

    # Convert types
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df["split_factor"] = pd.to_numeric(df["split_factor"], errors="coerce")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)

    # Handle nulls/blanks
    df = df.replace(
        to_replace=["None", "none", "NaN", "nan", ""],
        value=np.nan
    )
    df = df.infer_objects(copy=False)
    df = df.where(pd.notnull(df), None)

    # Drop duplicates just in case
    df.drop_duplicates(subset=["symbol", "effective_date"], inplace=True)

    return df

def standardize_dividends_columns(df):
    column_map = {
        "ex_dividend_date": "ex_dividend_date",
        "declaration_date": "declaration_date",
        "record_date": "record_date",
        "payment_date": "payment_date",
        "amount": "amount",
        "symbol": "symbol"
    }
    df = df.rename(columns=column_map)
    df["ex_dividend_date"] = pd.to_datetime(df["ex_dividend_date"], errors="coerce")
    df["declaration_date"] = pd.to_datetime(df["declaration_date"], errors="coerce")
    df["record_date"] = pd.to_datetime(df["record_date"], errors="coerce")
    df["payment_date"] = pd.to_datetime(df["payment_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)
    # Replace any "None", "NaN", "NaT" with np.nan (as a safety)
    df = df.replace(
        to_replace=["None", "none", "NaN", "nan", ""],
        value=np.nan
    )
    # Now, **force** any NaT or np.nan to Python None
    # Final fix before DB insert
    for col in ["ex_dividend_date", "declaration_date", "record_date", "payment_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").apply(lambda x: x.date() if pd.notnull(x) else None)
            df[col] = df[col].apply(lambda x: None if pd.isnull(x) or x is pd.NaT or x == "NaT" else x)
    # Drop duplicates if needed
    df.drop_duplicates(subset=["symbol", "ex_dividend_date"], inplace=True)
    return df
