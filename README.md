# For project planning and ideation go to Miro:
            https://miro.com/app/board/uXjVNeopges=/?share_link_id=828423247340


# General steps
## principles from the literature (Zvi Bodie, Kosztolanyi, Howard S. Marks: Mastering the market cycle)
## real life evidences about the principles from data analysis
## real time decisions on out of the box analitics systems
## data feed based automated decision system???

#### january plan
1. Explore alpha vantage capabilities. -> done
2. Download relevant data for one stock.
3. Create plotly graphs in a notebook for the analysis of all the downloaded data, or a dash EDA tool.
    ideas:
        - for comparison industry category averages could be plotted on alongside the base plot for cash flow and earnings
        - create aggreagte views for the company charts
4. Find patterns for trend reversals aiming for at least 20% change in a few month and dowload the news in their vicinity.
          -> iterate 4, 3, 2 and if necessary 1 in this order if the first attempt was not successful
5. If 4 was successfull do the above for 5 more stocks.
deadline January the 19th!

6. If 5 was successfull phrase it as an algorithm and test it on 5 other stocks!
deadline February the 2nd 


### Database structure
    1. Company fundamentals:
        - company base
        - daily OHLCV from IPO
        - corporate actions dividends and splits
        - income statement
        - balance sheet
        - cash flow
        - earnings per share EPS
    2. Company news database
        - as the first iteration it should only be based on the alpha vantage
    3. Economic fundamentals
        - just from alpha vantage at first
    4. Economic news
        - to be decided later


    updates should be done on:
        - company_base
        - corporate_actions
        - income_statement
        - balance_sheet
        - cash flow    
    when a company reports its latest earnings and financials!



