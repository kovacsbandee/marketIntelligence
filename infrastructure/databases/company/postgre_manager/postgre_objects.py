from sqlalchemy import Column, Integer, BigInteger, String, Float, Date, Sequence, DateTime, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class CompanyFundamentals(Base):
    """
    ORM osztály a vállalat alapvető pénzügyi mutatóinak tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat tőzsdei jelzése (pl.: AAPL az Apple-re, MSFT a Microsoft-ra).

    Mezők:
        - asset_type (String, nullable=True): Az eszköz típusa (pl.: Common Stock).
        - name (String, nullable=True): A vállalat neve.
        - description (String, nullable=True): A vállalat leírása.
        - cik (Integer, nullable=True): A vállalat CIK azonosítója.
        - exchange (String, nullable=True): A tőzsde, ahol az eszközt jegyzik (pl.: NYSE, NASDAQ).
        - currency (String, nullable=True): Az eszköz kereskedésére használt valuta (pl.: USD, EUR).
        - country (String, nullable=True): Az ország, ahol a vállalat alapvetően található.
        - sector (String, nullable=True): A vállalat iparági szektora (pl.: Technológia).
        - industry (String, nullable=True): A konkrét iparág (pl.: Szoftver).
        - address (String, nullable=True): A vállalat hivatalos címe.
        - official_site (String, nullable=True): A vállalat hivatalos weboldalának URL-je.
        - fiscal_year_end (String, nullable=True): A vállalat pénzügyi évének utolsó napja (pl.: December).
        - latest_quarter (Date, nullable=True): A legutóbbi pénzügyi negyedév jelentésének dátuma.
        - market_capitalization (Float, nullable=True): A vállalat piaci kapitalizációja.
        - ebitda (Float, nullable=True): EBITDA – adózás, kamatok, értékcsökkenés és amortizáció előtti eredmény.
        - pe_ratio (Float, nullable=True): Ár-nyereség arány (P/E ratio).
        - peg_ratio (Float, nullable=True): Ár-nyereség-növekedés arány (PEG ratio).
        - book_value (Float, nullable=True): A vállalat könyv szerinti értéke.
        - dividend_per_share (Float, nullable=True): Az egy részvényre fizetett osztalék.
        - dividend_yield (Float, nullable=True): Osztalékhozam százalékban.
        - eps (Float, nullable=True): Egy részvényre jutó nyereség (EPS).
        - revenue_per_share_ttm (Float, nullable=True): Egy részvényre jutó árbevétel az utolsó 12 hónap alapján.
        - profit_margin (Float, nullable=True): Nyereséghányad.
        - operating_margin_ttm (Float, nullable=True): Működési nyereség az utolsó 12 hónapban.
        - return_on_assets_ttm (Float, nullable=True): Eszközarányos megtérülés (ROA).
        - return_on_equity_ttm (Float, nullable=True): Sajáttőke-arányos megtérülés (ROE).
        - revenue_ttm (Float, nullable=True): Az utolsó 12 hónap teljes árbevétele.
        - gross_profit_ttm (Float, nullable=True): Az utolsó 12 hónap bruttó nyeresége.
        - diluted_eps_ttm (Float, nullable=True): Hígított egy részvényre jutó nyereség.
        - quarterly_earnings_growth_yoy (Float, nullable=True): Negyedéves nyereségnövekedés az előző év azonos időszakához képest.
        - quarterly_revenue_growth_yoy (Float, nullable=True): Negyedéves árbevétel-növekedés az előző év azonos időszakához képest.
        - analyst_target_price (Float, nullable=True): Elemzők által megcélzott ár.
        - analyst_rating_strong_buy (Integer, nullable=True): Erősen ajánlott vételi javaslatok száma.
        - analyst_rating_buy (Integer, nullable=True): Vételi javaslatok száma.
        - analyst_rating_hold (Integer, nullable=True): Megtartási javaslatok száma.
        - analyst_rating_sell (Integer, nullable=True): Eladási javaslatok száma.
        - analyst_rating_strong_sell (Integer, nullable=True): Erősen ajánlott eladási javaslatok száma.
        - trailing_pe (Float, nullable=True): Az ár-nyereség arány az elmúlt 12 hónapban.
        - forward_pe (Float, nullable=True): Az ár-nyereség arány a jövőbeli előrejelzett nyereség alapján.
        - price_to_sales_ratio_ttm (Float, nullable=True): Ár/Értékesítés arány az utolsó 12 hónap alapján.
        - price_to_book_ratio (Float, nullable=True): Ár/Könyv szerinti érték arány.
        - ev_to_revenue (Float, nullable=True): Vállalati érték osztva az árbevétellel.
        - ev_to_ebitda (Float, nullable=True): Vállalati érték osztva az EBITDA-val.
        - beta (Float, nullable=True): A részvény piaci volatilitása.
        - fifty_two_week_high (Float, nullable=True): Az elmúlt 52 hét legmagasabb árfolyama.
        - fifty_two_week_low (Float, nullable=True): Az elmúlt 52 hét legalacsonyabb árfolyama.
        - fifty_day_moving_average (Float, nullable=True): Az elmúlt 50 nap részvényárfolyamának átlaga.
        - two_hundred_day_moving_average (Float, nullable=True): Az elmúlt 200 nap részvényárfolyamának átlaga.
        - shares_outstanding (Integer, nullable=True): A kibocsátott részvények száma.
        - dividend_date (Date, nullable=True): Az osztalékfizetés dátuma.
        - ex_dividend_date (Date, nullable=True): Az osztalékra jogosító utolsó vásárlási dátum.

    Megjegyzés:
        - A `symbol` az elsődleges kulcs, mivel az egyedi azonosítást biztosítja.
        - Az összes pénzügyi mutató `NULL` értéket vehet fel, ha az adat nem áll rendelkezésre.
    """

    __tablename__ = 'company_fundamentals'

    symbol = Column(String, nullable=False, primary_key=True)
    asset_type = Column(String, nullable=True)
    name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    cik = Column(Integer, nullable=True)
    exchange = Column(String, nullable=True)
    currency = Column(String, nullable=True)
    country = Column(String, nullable=True)
    sector = Column(String, nullable=True)
    industry = Column(String, nullable=True)
    address = Column(String, nullable=True)
    official_site = Column(String, nullable=True)
    fiscal_year_end = Column(String, nullable=True)
    latest_quarter = Column(Date, nullable=True)
    market_capitalization = Column(Float, nullable=True)
    ebitda = Column(Float, nullable=True)
    pe_ratio = Column(Float, nullable=True)
    peg_ratio = Column(Float, nullable=True)
    book_value = Column(Float, nullable=True)
    dividend_per_share = Column(Float, nullable=True)
    dividend_yield = Column(Float, nullable=True)
    eps = Column(Float, nullable=True)
    revenue_per_share_ttm = Column(Float, nullable=True)
    profit_margin = Column(Float, nullable=True)
    operating_margin_ttm = Column(Float, nullable=True)
    return_on_assets_ttm = Column(Float, nullable=True)
    return_on_equity_ttm = Column(Float, nullable=True)
    revenue_ttm = Column(Float, nullable=True)
    gross_profit_ttm = Column(Float, nullable=True)
    diluted_eps_ttm = Column(Float, nullable=True)
    quarterly_earnings_growth_yoy = Column(Float, nullable=True)
    quarterly_revenue_growth_yoy = Column(Float, nullable=True)
    analyst_target_price = Column(Float, nullable=True)
    analyst_rating_strong_buy = Column(Integer, nullable=True)
    analyst_rating_buy = Column(Integer, nullable=True)
    analyst_rating_hold = Column(Integer, nullable=True)
    analyst_rating_sell = Column(Integer, nullable=True)
    analyst_rating_strong_sell = Column(Integer, nullable=True)
    trailing_pe = Column(Float, nullable=True)
    forward_pe = Column(Float, nullable=True)
    price_to_sales_ratio_ttm = Column(Float, nullable=True)
    price_to_book_ratio = Column(Float, nullable=True)
    ev_to_revenue = Column(Float, nullable=True)
    ev_to_ebitda = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    fifty_two_week_high = Column(Float, nullable=True)
    fifty_two_week_low = Column(Float, nullable=True)
    fifty_day_moving_average = Column(Float, nullable=True)
    two_hundred_day_moving_average = Column(Float, nullable=True)
    shares_outstanding = Column(BigInteger, nullable=True)
    dividend_date = Column(Date, nullable=True)
    ex_dividend_date = Column(Date, nullable=True)

    # --- Updater compatibility fields ---
    data_state = Column(String, nullable=True, default="unknown")  # e.g. "up_to_date", "outdated"
    last_update = Column(DateTime, nullable=True)


class DailyTimeSeries(Base):
    """
    ORM osztály a napi kereskedési adatok (candlestick) tárolására.

    Elsődleges kulcs:
        - date (Date): A kereskedési adatok dátuma (formátum: ÉÉÉÉ-HH-NN).
        - symbol (String): Az eszköz tőzsdei jelzése (például AAPL az Apple-re, TSLA a Tesla-ra).

    Mezők:
        - open (Float, nullable=True): Az eszköz nyitóára az adott kereskedési napon.
        - high (Float, nullable=True): Az eszköz által elért legmagasabb ár a kereskedési napon.
        - low (Float, nullable=True): Az eszköz által elért legalacsonyabb ár a kereskedési napon.
        - close (Float, nullable=True): Az eszköz záróára a kereskedési nap végén.
        - volume (Integer, nullable=True): Az adott kereskedési napon forgalmazott részvények összesített száma.

    Megjegyzés:
        - A `date` és `symbol` kombináció biztosítja az egyedi rekordokat.
        - A pénzügyi mezők (`open`, `high`, `low`, `close`, `volume`) `NULL` értéket vehetnek fel, ha az adatok nem elérhetők.
    """
    __tablename__ = 'daily_timeseries'

    date = Column(Date, nullable=False, primary_key=True)
    symbol = Column(String, nullable=False, primary_key=True)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    volume = Column(BigInteger, nullable=True)


class AnnualBalanceSheet(Base):
    """
    ORM osztály az éves mérlegek tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat tőzsdei jelzése (például AAPL az Apple-re, MSFT a Microsoft-ra).
        - fiscal_date_ending (Date): A pénzügyi év záródátuma, amelyre vonatkozóan a mérleget jelentik.

    Mezők:
        - reported_currency (String): Az a pénznem, amelyben a pénzügyi adatokat jelentették (például USD, EUR).
        - total_assets (Float): A vállalat tulajdonában lévő összes eszköz teljes értéke.
        - total_current_assets (Float): Azon eszközök összértéke, amelyek várhatóan készpénzzé alakulnak vagy felhasználásra kerülnek egy éven belül.
        - cash_and_cash_equivalents (Float, nullable=True): A vállalat által tartott készpénz és készpénzegyenértékesek értéke.
        - cash_and_short_term_investments (Float, nullable=True): A készpénz és rövid távú befektetések teljes értéke.
        - inventory (Float, nullable=True): Az áruk és anyagok értéke, amelyeket a vállalat értékesítésre vagy termelésre tart.
        - current_net_receivables (Float, nullable=True): A követelések teljes összege, a kétséges követelésekre képzett értékvesztéssel csökkentve.
        - total_non_current_assets (Float, nullable=True): A hosszú távú eszközök teljes értéke.
        - property_plant_equipment (Float, nullable=True): Az olyan fizikai eszközök bruttó értéke, mint például földterületek, épületek és gépek.
        - accumulated_depreciation_amortization_ppe (Float, nullable=True): Az ingatlanok, gépek és berendezések után felhalmozott értékcsökkenés.
        - intangible_assets (Float, nullable=True): Az immateriális javak teljes értéke.
        - goodwill (Float, nullable=True): A goodwill értéke, amely a megszerzett nettó eszközök valós értékét meghaladó prémiumot képviseli.
        - long_term_investments (Float, nullable=True): A befektetések értéke, amelyeket több mint egy évig szándékoznak megtartani.
        - short_term_investments (Float, nullable=True): Azon befektetések értéke, amelyek egy éven belül könnyen készpénzzé tehetők.
        - other_current_assets (Float, nullable=True): Az egy éven belül készpénzzé alakuló vagy felhasználható eszközök, amelyek nem tartoznak a főbb kategóriákba.
        - other_non_current_assets (Float, nullable=True): A hosszú távú eszközök értéke, amelyek nem tartoznak a főbb kategóriákba.
        - total_liabilities (Float): A vállalat összes kötelezettségének teljes összege.
        - total_current_liabilities (Float): Azon kötelezettségek teljes összege, amelyek egy éven belül esedékesek.
        - current_accounts_payable (Float, nullable=True): A szállítóknak fizetendő rövid távú kötelezettségek értéke.
        - deferred_revenue (Float, nullable=True): Az előre megkapott, de még nem teljesített bevételek értéke.
        - current_debt (Float, nullable=True): A rövid távú kölcsönök és egy éven belül esedékes adósságok összértéke.
        - short_term_debt (Float, nullable=True): Az egy éven belül esedékes adósságok értéke.
        - total_non_current_liabilities (Float): Azon kötelezettségek teljes összege, amelyek nem esedékesek egy éven belül.
        - capital_lease_obligations (Float, nullable=True): A tőkésített lízingekből eredő kötelezettségek teljes értéke.
        - long_term_debt (Float, nullable=True): Az egy éven túl esedékes adósságok értéke.
        - short_long_term_debt_total (Float, nullable=True): A rövid és hosszú távú adósságok teljes értéke.
        - other_current_liabilities (Float, nullable=True): Az egy éven belül esedékes egyéb kötelezettségek teljes értéke.
        - other_non_current_liabilities (Float, nullable=True): Az egy éven túl esedékes egyéb kötelezettségek teljes értéke.
        - total_shareholder_equity (Float): A vállalat nettó saját tőkéjének értéke.
        - treasury_stock (Float, nullable=True): A vállalat által visszavásárolt részvények értéke.
        - retained_earnings (Float, nullable=True): A vállalat által az osztalékok után megtartott nettó nyereség kumulatív összege.
        - common_stock (Float, nullable=True): A vállalat által kibocsátott közönséges részvények értéke.
        - common_stock_shares_outstanding (BigInteger, nullable=True): Az összes közönséges részvény száma, amely jelenleg a piacon forgalomban van.
    """
    __tablename__ = 'balance_sheet_annual'

    symbol = Column(String, primary_key=True, nullable=False)
    fiscal_date_ending = Column(Date, primary_key=True, nullable=False)
    reported_currency = Column(String, nullable=True)
    total_assets = Column(Float, nullable=True)
    total_current_assets = Column(Float, nullable=True)
    cash_and_cash_equivalents = Column(Float, nullable=True)
    cash_and_short_term_investments = Column(Float, nullable=True)
    inventory = Column(Float, nullable=True)
    current_net_receivables = Column(Float, nullable=True)
    total_non_current_assets = Column(Float, nullable=True)
    property_plant_equipment = Column(Float, nullable=True)
    accumulated_depreciation_amortization_ppe = Column(Float, nullable=True)
    intangible_assets = Column(Float, nullable=True)
    goodwill = Column(Float, nullable=True)
    long_term_investments = Column(Float, nullable=True)
    short_term_investments = Column(Float, nullable=True)
    other_current_assets = Column(Float, nullable=True)
    other_non_current_assets = Column(Float, nullable=True)
    total_liabilities = Column(Float, nullable=True)
    total_current_liabilities = Column(Float, nullable=True)
    current_accounts_payable = Column(Float, nullable=True)
    deferred_revenue = Column(Float, nullable=True)
    current_debt = Column(Float, nullable=True)
    short_term_debt = Column(Float, nullable=True)
    total_non_current_liabilities = Column(Float, nullable=True)
    capital_lease_obligations = Column(Float, nullable=True)
    long_term_debt = Column(Float, nullable=True)
    short_long_term_debt_total = Column(Float, nullable=True)
    other_current_liabilities = Column(Float, nullable=True)
    other_non_current_liabilities = Column(Float, nullable=True)
    total_shareholder_equity = Column(Float, nullable=True)
    treasury_stock = Column(Float, nullable=True)
    retained_earnings = Column(Float, nullable=True)
    common_stock = Column(Float, nullable=True)
    common_stock_shares_outstanding = Column(BigInteger, nullable=True)


class QuarterlyBalanceSheet(Base):
    """
    ORM osztály a vállalatok negyedéves mérlegadatainak tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat ticker szimbóluma (pl.: AAPL az Apple-hez, MSFT a Microsoft-hoz).
        - fiscal_date_ending (Date): A pénzügyi negyedév záródátuma, amelyre a mérleget jelentik.

    Mezők:
        - reported_currency (String): A pénzügyi adatok jelentésére használt valuta (pl.: USD, EUR).
        - total_assets (Float): A vállalat összes eszközének teljes értéke.
        - total_current_assets (Float): Azon eszközök összértéke, amelyeket egy éven belül várhatóan készpénzre váltanak vagy felhasználnak.
        - cash_and_cash_equivalents (Float, nullable=True): A vállalat által tartott készpénz és készpénz-egyenértékesek értéke.
        - cash_and_short_term_investments (Float, nullable=True): A készpénz és az olyan befektetések összértéke, amelyek rövid időn belül könnyen készpénzre válthatók.
        - inventory (Float, nullable=True): Azoknak az áruknak és anyagoknak az értéke, amelyeket a vállalat eladásra vagy gyártásra tart.
        - total_non_current_assets (Float, nullable=True): A hosszú távú eszközök összértéke, amelyeket nem várnak készpénzre váltani egy éven belül.
        - property_plant_equipment (Float, nullable=True): Az ingatlanok, gépek és berendezések bruttó értéke.
        - accumulated_depreciation_amortization_ppe (Float, nullable=True): Az ingatlanok, gépek és berendezések halmozott értékcsökkenése és amortizációja.
        - current_net_receivables (Float, nullable=True): A vevőktől egy éven belül várhatóan befolyó követelések teljes értéke, levonva a kétes követelésekre képzett céltartalékokat.
        - intangible_assets (Float, nullable=True): Az immateriális javak teljes értéke, például szabadalmak és védjegyek, beleértve a goodwillt is.
        - intangible_assets_excluding_goodwill (Float, nullable=True): Az immateriális javak értéke a goodwill kivételével.
        - goodwill (Float, nullable=True): A goodwill értéke, amely az akvirált nettó eszközök valós értékén felüli prémiumot jelenti.
        - investments (Float, nullable=True): A vállalat befektetéseinek teljes értéke.
        - long_term_investments (Float, nullable=True): Azoknak a befektetéseknek az értéke, amelyeket több mint egy éven keresztül tartanak.
        - short_term_investments (Float, nullable=True): Azon befektetések értéke, amelyek könnyen likvidálhatók egy éven belül.
        - other_current_assets (Float, nullable=True): Az egy éven belül készpénzre váltható vagy felhasználható egyéb eszközök értéke.
        - other_non_current_assets (Float, nullable=True): Az egy éven túl felhasználható hosszú távú eszközök értéke, amelyek nem tartoznak a főbb kategóriákba.
        - total_liabilities (Float): A vállalat által vállalt kötelezettségek teljes összege.
        - total_current_liabilities (Float): Az egy éven belül esedékes kötelezettségek teljes összege.
        - current_accounts_payable (Float, nullable=True): Az árukért és szolgáltatásokért beszállítóknak járó rövid távú kötelezettségek értéke.
        - deferred_revenue (Float, nullable=True): Az a bevétel, amelyet már megkaptak, de még nem kerestek meg (pl.: előlegek).
        - current_debt (Float, nullable=True): Az egy éven belül esedékes rövid távú hitelek és adósságok összértéke.
        - short_term_debt (Float, nullable=True): Az egy éven belül esedékes adósságkötelezettségek értéke.
        - total_non_current_liabilities (Float): Az egy éven túl esedékes kötelezettségek teljes összege.
        - capital_lease_obligations (Float, nullable=True): A tőkebérleti kötelezettségek teljes értéke.
        - long_term_debt (Float, nullable=True): Az egy éven túl esedékes adósságkötelezettségek értéke.
        - current_long_term_debt (Float, nullable=True): Az egy éven belül esedékes hosszú távú adósság rész.
        - long_term_debt_noncurrent (Float, nullable=True): Az egy éven túl esedékes hosszú távú adósság rész.
        - short_long_term_debt_total (Float, nullable=True): A rövid és hosszú távú adósságok teljes összege.
        - other_current_liabilities (Float, nullable=True): Az egy éven belül esedékes egyéb kötelezettségek teljes összege.
        - other_non_current_liabilities (Float, nullable=True): Az egy éven túl esedékes egyéb kötelezettségek teljes összege.
        - total_shareholder_equity (Float): A vállalat saját tőkéjének nettó értéke.
        - treasury_stock (Float, nullable=True): A vállalat által visszavásárolt és kincstári részvényként tartott részvények értéke.
        - retained_earnings (Float, nullable=True): A vállalat által az osztalékok kifizetése után megtartott nettó jövedelem halmozott összege.
        - common_stock (Float, nullable=True): A vállalat által kibocsátott törzsrészvények értéke.
        - common_stock_shares_outstanding (BigInteger, nullable=True): A piacon jelenleg forgalomban lévő törzsrészvények száma.
    """
    __tablename__ = 'balance_sheet_quarterly'

    symbol = Column(String, primary_key=True, nullable=False)
    fiscal_date_ending = Column(Date, primary_key=True, nullable=False)
    reported_currency = Column(String, nullable=True)
    total_assets = Column(Float, nullable=True)
    total_current_assets = Column(Float, nullable=True)
    cash_and_cash_equivalents = Column(Float, nullable=True)
    cash_and_short_term_investments = Column(Float, nullable=True)
    inventory = Column(Float, nullable=True)
    total_non_current_assets = Column(Float, nullable=True)
    property_plant_equipment = Column(Float, nullable=True)
    accumulated_depreciation_amortization_ppe = Column(Float, nullable=True)
    current_net_receivables = Column(Float, nullable=True)
    intangible_assets = Column(Float, nullable=True)
    intangible_assets_excluding_goodwill = Column(Float, nullable=True)
    goodwill = Column(Float, nullable=True)
    long_term_investments = Column(Float, nullable=True)
    short_term_investments = Column(Float, nullable=True)
    total_liabilities = Column(Float, nullable=True)
    total_current_liabilities = Column(Float, nullable=True)
    total_shareholder_equity = Column(Float, nullable=True)
    common_stock_shares_outstanding = Column(BigInteger, nullable=True)


class AnnualCashFlow(Base):
    """
    ORM osztály a vállalatok pénzforgalmi kimutatásainak tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat tőzsdei jelzése (például AAPL az Apple-re, MSFT a Microsoft-ra).
        - fiscal_date_ending (Date): A pénzügyi év záródátuma, amelyre vonatkozóan a cash flow adatokat jelentik.

    Mezők:
        - reported_currency (String): Az a pénznem, amelyben a cash flow adatokat jelentették (például USD, EUR).
        - operating_cashflow (Float): Az üzleti műveletek során generált vagy felhasznált nettó készpénz a pénzügyi év során.
        - payments_for_operating_activities (Float, nullable=True): Az üzleti tevékenységekre fordított készpénzkifizetések.
        - proceeds_from_operating_activities (Float, nullable=True): Az üzleti tevékenységekből származó készpénzbevételek.
        - change_in_operating_liabilities (Float, nullable=True): Az üzleti tevékenységekkel kapcsolatos kötelezettségek változása.
        - change_in_operating_assets (Float, nullable=True): Az üzleti tevékenységekkel kapcsolatos eszközök változása.
        - depreciation_depletion_and_amortization (Float, nullable=True): Az értékcsökkenés, kimerülés és amortizáció összesített értéke.
        - capital_expenditures (Float, nullable=True): Az ingatlanok, gyárak és berendezések fejlesztésére fordított készpénz (CapEx).
        - change_in_receivables (Float, nullable=True): A vállalat vevőköveteléseinek változása.
        - change_in_inventory (Float, nullable=True): A vállalat készleteinek értékváltozása.
        - profit_loss (Float): A pénzügyi évre jelentett nettó nyereség vagy veszteség.
        - cashflow_from_investment (Float, nullable=True): Befektetési tevékenységekből származó nettó készpénz.
        - cashflow_from_financing (Float, nullable=True): Finanszírozási tevékenységekből származó nettó készpénz.
        - proceeds_from_repayments_of_short_term_debt (Float, nullable=True): Rövid távú adósság törlesztéséhez kapcsolódó készpénzbevételek vagy -kiadások.
        - payments_for_repurchase_of_common_stock (Float, nullable=True): A vállalat közönséges részvényeinek visszavásárlására fordított készpénz.
        - payments_for_repurchase_of_equity (Float, nullable=True): A vállalat saját tőkéjének visszavásárlására fordított készpénz.
        - payments_for_repurchase_of_preferred_stock (Float, nullable=True): Az előnyös részvények visszavásárlására fordított készpénz.
        - dividend_payout (Float, nullable=True): A vállalat által osztalékként kifizetett készpénz.
        - dividend_payout_common_stock (Float, nullable=True): A közönséges részvényeseknek kifizetett osztalék.
        - dividend_payout_preferred_stock (Float, nullable=True): Az előnyös részvényeseknek kifizetett osztalék.
        - proceeds_from_issuance_of_common_stock (Float, nullable=True): Új közönséges részvények kibocsátásából származó készpénz.
        - proceeds_from_issuance_of_long_term_debt_and_capital_securities_net (Float, nullable=True): A hosszú távú adósság és tőkepiaci értékpapírok kibocsátásából származó nettó készpénz.
        - proceeds_from_issuance_of_preferred_stock (Float, nullable=True): Új előnyös részvények kibocsátásából származó készpénz.
        - proceeds_from_repurchase_of_equity (Float, nullable=True): A visszavásárolt saját tőke értékesítéséből származó készpénz.
        - proceeds_from_sale_of_treasury_stock (Float, nullable=True): A kincstári részvények eladásából származó készpénz.
        - change_in_cash_and_cash_equivalents (Float, nullable=True): A készpénz- és készpénzegyenértékesek nettó változása.
        - change_in_exchange_rate (Float, nullable=True): Az árfolyamváltozások hatása a készpénzre és készpénzegyenértékesekre.
        - net_income (Float): A vállalat nettó jövedelme (vagy vesztesége) a pénzügyi év során.

    Megjegyzés:
        - A `symbol` és `fiscal_date_ending` kombináció biztosítja az egyedi rekordokat.
        - Bizonyos pénzügyi mezők lehetnek `NULL`, mivel ezek az adatok nem mindig állnak rendelkezésre.
    """
    __tablename__ = 'cash_flow_annual'

    symbol = Column(String, primary_key=True, nullable=False)
    fiscal_date_ending = Column(Date, primary_key=True, nullable=False)
    reported_currency = Column(String, nullable=True)
    operating_cashflow = Column(Float, nullable=True)
    payments_for_operating_activities = Column(Float, nullable=True)
    proceeds_from_operating_activities = Column(Float, nullable=True)
    change_in_operating_liabilities = Column(Float, nullable=True)
    change_in_operating_assets = Column(Float, nullable=True)
    depreciation_depletion_and_amortization = Column(Float, nullable=True)
    capital_expenditures = Column(Float, nullable=True)
    change_in_receivables = Column(Float, nullable=True)
    change_in_inventory = Column(Float, nullable=True)
    profit_loss = Column(Float, nullable=True)
    cashflow_from_investment = Column(Float, nullable=True)
    cashflow_from_financing = Column(Float, nullable=True)
    proceeds_from_repayments_of_short_term_debt = Column(Float, nullable=True)
    payments_for_repurchase_of_common_stock = Column(Float, nullable=True)
    payments_for_repurchase_of_equity = Column(Float, nullable=True)
    payments_for_repurchase_of_preferred_stock = Column(Float, nullable=True)
    dividend_payout = Column(Float, nullable=True)
    dividend_payout_common_stock = Column(Float, nullable=True)
    dividend_payout_preferred_stock = Column(Float, nullable=True)
    proceeds_from_issuance_of_common_stock = Column(Float, nullable=True)
    proceeds_from_issuance_of_long_term_debt_and_capital_securities_net = Column(
        Float, nullable=True)
    proceeds_from_issuance_of_preferred_stock = Column(Float, nullable=True)
    proceeds_from_repurchase_of_equity = Column(Float, nullable=True)
    proceeds_from_sale_of_treasury_stock = Column(Float, nullable=True)
    change_in_cash_and_cash_equivalents = Column(Float, nullable=True)
    change_in_exchange_rate = Column(Float, nullable=True)
    net_income = Column(Float, nullable=True)


class QuarterlyCashFlow(Base):
    """
    ORM osztály a vállalatok negyedéves pénzforgalmi kimutatásainak tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat tőzsdei jelzése (például AAPL az Apple-re, MSFT a Microsoft-ra).
        - fiscal_date_ending (Date): A pénzügyi negyedév záródátuma, amelyre vonatkozóan a cash flow adatokat jelentik.

    Mezők:
        - reported_currency (String): Az a pénznem, amelyben a cash flow adatokat jelentették (például USD, EUR).
        - operating_cashflow (Float): Az üzleti műveletek során generált vagy felhasznált nettó készpénz a negyedév során.
        - payments_for_operating_activities (Float, nullable=True): Az üzleti tevékenységekre fordított készpénzkifizetések.
        - proceeds_from_operating_activities (Float, nullable=True): Az üzleti tevékenységekből származó készpénzbevételek.
        - change_in_operating_liabilities (Float, nullable=True): Az üzleti tevékenységekkel kapcsolatos kötelezettségek változása.
        - change_in_operating_assets (Float, nullable=True): Az üzleti tevékenységekkel kapcsolatos eszközök változása.
        - depreciation_depletion_and_amortization (Float, nullable=True): Az értékcsökkenés, kimerülés és amortizáció összesített értéke.
        - capital_expenditures (Float, nullable=True): Az ingatlanok, gyárak és berendezések fejlesztésére fordított készpénz (CapEx).
        - change_in_receivables (Float, nullable=True): A vállalat vevőköveteléseinek változása.
        - change_in_inventory (Float, nullable=True): A vállalat készleteinek negyedéves értékváltozása.
        - profit_loss (Float): A negyedévre jelentett nettó nyereség vagy veszteség.
        - cashflow_from_investment (Float, nullable=True): Befektetési tevékenységekből származó nettó készpénz.
        - cashflow_from_financing (Float, nullable=True): Finanszírozási tevékenységekből származó nettó készpénz.
        - proceeds_from_repayments_of_short_term_debt (Float, nullable=True): Rövid távú adósság törlesztéséhez kapcsolódó készpénzbevételek vagy -kiadások.
        - payments_for_repurchase_of_common_stock (Float, nullable=True): A vállalat közönséges részvényeinek visszavásárlására fordított készpénz.
        - payments_for_repurchase_of_equity (Float, nullable=True): A vállalat saját tőkéjének visszavásárlására fordított készpénz.
        - payments_for_repurchase_of_preferred_stock (Float, nullable=True): Az előnyös részvények visszavásárlására fordított készpénz.
        - dividend_payout (Float, nullable=True): A vállalat által osztalékként kifizetett készpénz.
        - dividend_payout_common_stock (Float, nullable=True): A közönséges részvényeseknek kifizetett osztalék.
        - dividend_payout_preferred_stock (Float, nullable=True): Az előnyös részvényeseknek kifizetett osztalék.
        - proceeds_from_issuance_of_common_stock (Float, nullable=True): Új közönséges részvények kibocsátásából származó készpénz.
        - proceeds_from_issuance_of_long_term_debt_and_capital_securities_net (Float, nullable=True): A hosszú távú adósság és tőkepiaci értékpapírok kibocsátásából származó nettó készpénz.
        - proceeds_from_issuance_of_preferred_stock (Float, nullable=True): Új előnyös részvények kibocsátásából származó készpénz.
        - proceeds_from_repurchase_of_equity (Float, nullable=True): A visszavásárolt saját tőke értékesítéséből származó készpénz.
        - proceeds_from_sale_of_treasury_stock (Float, nullable=True): A kincstári részvények eladásából származó készpénz.
        - change_in_cash_and_cash_equivalents (Float, nullable=True): A készpénz- és készpénzegyenértékesek nettó változása.
        - change_in_exchange_rate (Float, nullable=True): Az árfolyamváltozások hatása a készpénzre és készpénzegyenértékesekre.
        - net_income (Float): A vállalat nettó jövedelme (vagy vesztesége) a negyedévre vonatkozóan.

    Megjegyzés:
        - A `symbol` és `fiscal_date_ending` kombináció biztosítja az egyedi rekordokat.
        - Bizonyos pénzügyi mezők lehetnek `NULL`, mivel ezek az adatok nem mindig állnak rendelkezésre.
    """
    __tablename__ = 'cash_flow_quarterly'

    symbol = Column(String, primary_key=True, nullable=False)
    fiscal_date_ending = Column(Date, primary_key=True, nullable=False)
    reported_currency = Column(String, nullable=True)
    operating_cashflow = Column(Float, nullable=True)
    payments_for_operating_activities = Column(Float, nullable=True)
    proceeds_from_operating_activities = Column(Float, nullable=True)
    change_in_operating_liabilities = Column(Float, nullable=True)
    change_in_operating_assets = Column(Float, nullable=True)
    depreciation_depletion_and_amortization = Column(Float, nullable=True)
    capital_expenditures = Column(Float, nullable=True)
    change_in_receivables = Column(Float, nullable=True)
    change_in_inventory = Column(Float, nullable=True)
    profit_loss = Column(Float, nullable=True)
    cashflow_from_investment = Column(Float, nullable=True)
    cashflow_from_financing = Column(Float, nullable=True)
    proceeds_from_repayments_of_short_term_debt = Column(Float, nullable=True)
    payments_for_repurchase_of_common_stock = Column(Float, nullable=True)
    payments_for_repurchase_of_equity = Column(Float, nullable=True)
    payments_for_repurchase_of_preferred_stock = Column(Float, nullable=True)
    dividend_payout = Column(Float, nullable=True)
    dividend_payout_common_stock = Column(Float, nullable=True)
    dividend_payout_preferred_stock = Column(Float, nullable=True)
    proceeds_from_issuance_of_common_stock = Column(Float, nullable=True)
    proceeds_from_issuance_of_long_term_debt_and_capital_securities_net = Column(
        Float, nullable=True)
    proceeds_from_issuance_of_preferred_stock = Column(Float, nullable=True)
    proceeds_from_repurchase_of_equity = Column(Float, nullable=True)
    proceeds_from_sale_of_treasury_stock = Column(Float, nullable=True)
    change_in_cash_and_cash_equivalents = Column(Float, nullable=True)
    change_in_exchange_rate = Column(Float, nullable=True)
    net_income = Column(Float, nullable=True)


class AnnualEarnings(Base):
    """
    ORM osztály a vállalatok éves eredményjelentéseinek tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat tőzsdei jelzése (például AAPL az Apple-re, MSFT a Microsoft-ra).
        - fiscal_date_ending (Date): A pénzügyi év záródátuma, amelyre vonatkozóan a nyereséget jelentik.

    Mezők:
        - reported_eps (Float): Az egy részvényre jutó eredmény (EPS), amelyet a vállalat jelentett a pénzügyi évre.

    Megjegyzés:
        - A `symbol` és `fiscal_date_ending` kombináció biztosítja az egyedi rekordokat.
        - Az `reported_eps` oszlop az adott pénzügyi év teljes éves EPS értékét tartalmazza.
    """
    __tablename__ = 'earnings_annual'

    symbol = Column(String, primary_key=True, nullable=False)
    fiscal_date_ending = Column(Date, primary_key=True, nullable=False)
    reported_eps = Column(Float, nullable=True)


class QuarterlyEarnings(Base):
    """
    ORM osztály a vállalatok negyedéves eredményjelentéseinek tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat ticker szimbóluma (pl.: AAPL Apple-hez, MSFT Microsoft-hoz).
        - fiscal_date_ending (Date): A jelentett pénzügyi negyedév vége (pl.: 2024-09-30).

    Mezők:
        - reported_date (Date): A tényleges dátum, amikor a nyereséget nyilvánosságra hozták.
        - reported_eps (Float): A vállalat által a negyedévre jelentett egy részvényre jutó nyereség (EPS).
        - estimated_eps (Float, nullable=True): Az elemzők által becsült egy részvényre jutó nyereség (EPS) az adott negyedévre.
        - surprise (Float, nullable=True): A jelentett EPS és a becsült EPS közötti különbség (kiszámítása: reportedEPS - estimatedEPS).
        - surprise_percentage (Float, nullable=True): A jelentett EPS és a becsült EPS közötti százalékos eltérés (kiszámítása: (surprise / estimatedEPS) * 100).
        - report_time (String): A jelentés közzétételének időpontja, általában 'BTO' (tőzsdenyitás előtt) vagy 'AMC' (piaczárás után).

    Megjegyzés:
        - A `symbol` és `fiscal_date_ending` kombináció biztosítja az egyedi rekordokat.
        - Az `estimated_eps`, `surprise`, és `surprise_percentage` értékei lehetnek nullák, ha az adott negyedévhez nincs becsült adat.
    """
    __tablename__ = 'earnings_quarterly'

    symbol = Column(String, primary_key=True, nullable=False)
    fiscal_date_ending = Column(Date, primary_key=True, nullable=False)
    reported_date = Column(Date, nullable=True)
    reported_eps = Column(Float, nullable=True)
    estimated_eps = Column(Float, nullable=True)
    surprise = Column(Float, nullable=True)
    surprise_percentage = Column(Float, nullable=True)
    report_time = Column(String, nullable=True)


class AnnualIncomeStatement(Base):
    """
    ORM osztály a vállalatok éves eredménykimutatásainak tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat tőzsdei jelzése (például AAPL az Apple-re, MSFT a Microsoft-ra).
        - fiscal_date_ending (Date): A pénzügyi év záródátuma, amelyre vonatkozóan az eredménykimutatást jelentik.

    Mezők:
        - reported_currency (String): Az a pénznem, amelyben az eredménykimutatás adatai jelentve vannak (például USD, EUR).
        - gross_profit (Float): Az összes bevétel mínusz az eladott áruk költsége (COGS), amely a vállalat nyereségét jelzi a közvetlen költségek levonása után.
        - total_revenue (Float): A vállalat által a pénzügyi év során generált teljes bevétel, beleértve minden értékesítést és egyéb bevételt.
        - cost_of_revenue (Float): A bevétel előállításához kapcsolódó összköltség, beleértve a gyártási és szállítási költségeket.
        - cost_of_goods_and_services_sold (Float): Az eladott áruk vagy szolgáltatások előállításának közvetlen költségei.
        - operating_income (Float): A vállalat alaptevékenységéből származó nyereség, amelyet a bruttó nyereségből az üzemi költségek levonásával számítanak ki.
        - selling_general_and_administrative (Float): Az értékesítéssel, működéssel és adminisztratív tevékenységekkel kapcsolatos költségek.
        - research_and_development (Float): A kutatási és fejlesztési tevékenységekre fordított költségek a pénzügyi év során.
        - operating_expenses (Float): A normál üzleti tevékenységekből származó összes költség, kivéve az árbevétel költségét.
        - investment_income_net (Float): A befektetésekből származó nettó bevétel, például kamatok vagy osztalékok, levonva a kapcsolódó költségeket.
        - net_interest_income (Float): A kamatokkal kapcsolatos tevékenységekből származó nettó bevétel, amelyet a kamatbevétel és kamatköltség különbségeként számítanak ki.
        - interest_income (Float): A befektetésekből vagy kölcsönökből származó kamatbevétel.
        - interest_expense (Float): Az adósságokra vagy kölcsönökre eső kamatköltségek a pénzügyi év során.
        - non_interest_income (Float): A nem kamatjellegű tevékenységekből származó bevétel, például díjak vagy jutalékok.
        - other_non_operating_income (Float): Nem üzemi tevékenységekből származó bevétel, például egyszeri nyereségek vagy szokatlan tételek.
        - depreciation (Float): A tárgyi eszközök költségeinek allokációja a használati időszakuk alatt a pénzügyi év során.
        - depreciation_and_amortization (Float): Az összes nem készpénzes költség, amely a tárgyi eszközök értékcsökkenéséből és az immateriális javak amortizációjából származik.
        - income_before_tax (Float): A vállalat jövedelme az adókiadások levonása előtt.
        - income_tax_expense (Float): Az adókiadások teljes összege, amelyet a vállalat a pénzügyi év során viselt.
        - interest_and_debt_expense (Float): Az adósságok és egyéb kölcsönzési költségek kapcsán felmerült összes költség.
        - net_income_from_continuing_operations (Float): A normál, rendszeres üzleti tevékenységekből származó nettó jövedelem.
        - comprehensive_income_net_of_tax (Float): A vállalat teljes jövedelme, beleértve a nettó jövedelmet és az egyéb átfogó jövedelemtételeket az adók után.
        - ebit (Float): Earnings Before Interest and Taxes (EBIT) – egy nyereségességi mutató, amely kizárja a kamat- és adóköltségeket.
        - ebitda (Float): Earnings Before Interest, Taxes, Depreciation, and Amortization (EBITDA) – egy üzemi teljesítményt mérő mutató.
        - net_income (Float): A vállalat teljes nyeresége vagy vesztesége, miután minden költséget, adót és kiadást levontak a bevételből.
    """
    __tablename__ = 'income_statement_annual'

    symbol = Column(String, primary_key=True, nullable=False)
    fiscal_date_ending = Column(Date, primary_key=True, nullable=False)
    reported_currency = Column(String, nullable=True)
    gross_profit = Column(Float, nullable=True)
    total_revenue = Column(Float, nullable=True)
    cost_of_revenue = Column(Float, nullable=True)
    cost_of_goods_and_services_sold = Column(Float, nullable=True)
    operating_income = Column(Float, nullable=True)
    selling_general_and_administrative = Column(Float, nullable=True)
    research_and_development = Column(Float, nullable=True)
    operating_expenses = Column(Float, nullable=True)
    investment_income_net = Column(Float, nullable=True)
    net_interest_income = Column(Float, nullable=True)
    interest_income = Column(Float, nullable=True)
    interest_expense = Column(Float, nullable=True)
    non_interest_income = Column(Float, nullable=True)
    other_non_operating_income = Column(Float, nullable=True)
    depreciation = Column(Float, nullable=True)
    depreciation_and_amortization = Column(Float, nullable=True)
    income_before_tax = Column(Float, nullable=True)
    income_tax_expense = Column(Float, nullable=True)
    interest_and_debt_expense = Column(Float, nullable=True)
    net_income_from_continuing_operations = Column(Float, nullable=True)
    comprehensive_income_net_of_tax = Column(Float, nullable=True)
    ebit = Column(Float, nullable=True)
    ebitda = Column(Float, nullable=True)
    net_income = Column(Float, nullable=True)


class QuarterlyIncomeStatement(Base):
    """
    ORM osztály a vállalatok negyedéves eredménykimutatásainak tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat tőzsdei jelzése (például AAPL az Apple-re, MSFT a Microsoft-ra).
        - fiscal_date_ending (Date): A pénzügyi negyedév záródátuma, amelyre vonatkozóan az eredménykimutatást jelentik.

    Mezők:
        - reported_currency (String): Az a pénznem, amelyben az eredménykimutatás adatai jelentve vannak (például USD, EUR).
        - gross_profit (Float): Az összes bevétel mínusz az eladott áruk költsége (COGS), amely a vállalat nyereségét jelzi a közvetlen költségek levonása után.
        - total_revenue (Float): A vállalat által a negyedév során generált teljes bevétel, beleértve minden értékesítést és egyéb bevételt.
        - cost_of_revenue (Float): A bevétel előállításához kapcsolódó összköltség, beleértve a gyártási és szállítási költségeket.
        - cost_of_goods_and_services_sold (Float): Az eladott áruk vagy szolgáltatások előállításának közvetlen költségei.
        - operating_income (Float): A vállalat alaptevékenységéből származó nyereség, amelyet a bruttó nyereségből az üzemi költségek levonásával számítanak ki.
        - selling_general_and_administrative (Float): Az értékesítéssel, működéssel és adminisztratív tevékenységekkel kapcsolatos költségek.
        - research_and_development (Float): A kutatási és fejlesztési tevékenységekre fordított költségek a negyedév során.
        - operating_expenses (Float): A normál üzleti tevékenységekből származó összes költség, kivéve az árbevétel költségét.
        - investment_income_net (Float): A befektetésekből származó nettó bevétel, például kamatok vagy osztalékok, levonva a kapcsolódó költségeket.
        - net_interest_income (Float): A kamatokkal kapcsolatos tevékenységekből származó nettó bevétel, amelyet a kamatbevétel és kamatköltség különbségeként számítanak ki.
        - interest_income (Float): A befektetésekből vagy kölcsönökből származó kamatbevétel.
        - interest_expense (Float): Az adósságokra vagy kölcsönökre eső kamatköltségek a negyedév során.
        - non_interest_income (Float): A nem kamatjellegű tevékenységekből származó bevétel, például díjak vagy jutalékok.
        - other_non_operating_income (Float): Nem üzemi tevékenységekből származó bevétel, például egyszeri nyereségek vagy szokatlan tételek.
        - depreciation (Float): A tárgyi eszközök költségeinek allokációja a használati időszakuk alatt a negyedév során.
        - depreciation_and_amortization (Float): Az összes nem készpénzes költség, amely a tárgyi eszközök értékcsökkenéséből és az immateriális javak amortizációjából származik.
        - income_before_tax (Float): A vállalat jövedelme az adókiadások levonása előtt.
        - income_tax_expense (Float): Az adókiadások teljes összege, amelyet a vállalat a negyedév során viselt.
        - interest_and_debt_expense (Float): Az adósságok és egyéb kölcsönzési költségek kapcsán felmerült összes költség.
        - net_income_from_continuing_operations (Float): A normál, rendszeres üzleti tevékenységekből származó nettó jövedelem.
        - comprehensive_income_net_of_tax (Float): A vállalat teljes jövedelme, beleértve a nettó jövedelmet és az egyéb átfogó jövedelemtételeket az adók után.
        - ebit (Float): Earnings Before Interest and Taxes (EBIT) – egy nyereségességi mutató, amely kizárja a kamat- és adóköltségeket.
        - ebitda (Float): Earnings Before Interest, Taxes, Depreciation, and Amortization (EBITDA) – egy üzemi teljesítményt mérő mutató.
        - net_income (Float): A vállalat teljes nyeresége vagy vesztesége, miután minden költséget, adót és kiadást levontak a bevételből.
    """
    __tablename__ = 'income_statement_quarterly'

    symbol = Column(String, primary_key=True, nullable=False)
    fiscal_date_ending = Column(Date, primary_key=True, nullable=False)
    reported_currency = Column(String, nullable=True)
    gross_profit = Column(Float, nullable=True)
    total_revenue = Column(Float, nullable=True)
    cost_of_revenue = Column(Float, nullable=True)
    cost_of_goods_and_services_sold = Column(Float, nullable=True)
    operating_income = Column(Float, nullable=True)
    selling_general_and_administrative = Column(Float, nullable=True)
    research_and_development = Column(Float, nullable=True)
    operating_expenses = Column(Float, nullable=True)
    investment_income_net = Column(Float, nullable=True)
    net_interest_income = Column(Float, nullable=True)
    interest_income = Column(Float, nullable=True)
    interest_expense = Column(Float, nullable=True)
    non_interest_income = Column(Float, nullable=True)
    other_non_operating_income = Column(Float, nullable=True)
    depreciation = Column(Float, nullable=True)
    depreciation_and_amortization = Column(Float, nullable=True)
    income_before_tax = Column(Float, nullable=True)
    income_tax_expense = Column(Float, nullable=True)
    interest_and_debt_expense = Column(Float, nullable=True)
    net_income_from_continuing_operations = Column(Float, nullable=True)
    comprehensive_income_net_of_tax = Column(Float, nullable=True)
    ebit = Column(Float, nullable=True)
    ebitda = Column(Float, nullable=True)
    net_income = Column(Float, nullable=True)


class InsiderTransactions(Base):
    """
    InsiderTransactions osztály, amely az "insider_transactions" adatbázis táblát reprezentálja.

    Attribútumok:
        transaction_date (datetime.date): Az ügylet dátuma.
        symbol (str): A részvény szimbóluma.
        executive (str): Az ügyletet végrehajtó vezető neve.
        executive_title (str): A vezető pozíciója.
        security_type (str): Az értékpapír típusa.
        acquisition_or_disposal (str): Az ügylet típusa (vásárlás vagy eladás).
        shares (float): Az ügyletben érintett részvények száma.
        share_price (float): Az egy részvényre jutó ár.
    """
    __tablename__ = "insider_transactions"
    __table_args__ = (
        PrimaryKeyConstraint(
            "transaction_date",
            "symbol",
            "executive",
            "executive_title",
            "security_type",
            "acquisition_or_disposal",
            name="insider_transactions_pkey"
        ),
    )

    transaction_date = Column(Date, nullable=False)
    symbol = Column(String, nullable=False)
    executive = Column(String, nullable=False)
    executive_title = Column(String, nullable=True)
    security_type = Column(String, nullable=False)
    acquisition_or_disposal = Column(String, nullable=False)
    shares = Column(Float, nullable=True)
    share_price = Column(Float, nullable=True)


class StockSplit(Base):
    """
    ORM osztály a részvényfelosztások tárolására.

    Elsődleges kulcs:
        - symbol (String): A részvényfelosztást végrehajtó vállalat tőzsdei jelzése (például AAPL az Apple-re, MSFT a Microsoft-ra).
        - effective_date (Date): Az a dátum, amikor a részvényfelosztás hatályba lép.

    Mezők:
        - split_factor (Float): A részvényfelosztás aránya, amely megmutatja, hogyan oszlanak meg a részvények (például 2.0 egy kettő az egyhez felosztás esetén).
    """
    __tablename__ = 'stock_splits'

    symbol = Column(String, primary_key=True, nullable=False)
    effective_date = Column(Date, primary_key=True, nullable=False)
    split_factor = Column(Float, nullable=True)


class Dividends(Base):
    """
    ORM osztály a vállalatok által fizetett osztalékok tárolására.

    Elsődleges kulcs:
        - symbol (String): A vállalat ticker szimbóluma, amely az osztalékot kibocsátja (pl.: AAPL az Apple-hez, MSFT a Microsoft-hoz).
        - ex_dividend_date (Date): Az a dátum, ameddig a részvényt birtokolni kell ahhoz, hogy az osztalékra jogosult legyen.

    Mezők:
        - amount (Float, nullable=True): Az egy részvényre jutó osztalék összege, amelyet a részvényeseknek kifizetnek.
        - declaration_date (Date, nullable=True): Az a dátum, amikor a vállalat igazgatótanácsa hivatalosan bejelenti az osztalékot.
        - record_date (Date, nullable=True): Az a dátum, amikor a vállalat felülvizsgálja nyilvántartásait az osztalékra jogosult részvényesek meghatározása érdekében.
        - payment_date (Date, nullable=True): Az a dátum, amikor az osztalékot kifizetik a részvényeseknek.

    Megjegyzés:
        - A `symbol` és `ex_dividend_date` kombináció biztosítja az egyedi rekordokat.
        - Az `amount` mező nem lehet `NULL`, mivel minden osztalékkifizetéshez egy összeg tartozik.
        - A `declaration_date`, `record_date` és `payment_date` opcionális, mert ezek az információk nem mindig állnak rendelkezésre.
    """
    __tablename__ = 'dividends'
    # __table_args__ = (
    #     PrimaryKeyConstraint("symbol", "ex_dividend_date", name="dividends_pkey"),
    # )

    symbol = Column(String, primary_key=True,  nullable=False)
    ex_dividend_date = Column(Date, primary_key=True, nullable=False)
    amount = Column(Float, nullable=True)
    declaration_date = Column(Date, nullable=True)
    record_date = Column(Date, nullable=True)
    payment_date = Column(Date, nullable=True)


# --- Table name to ORM class mapping ---
table_name_to_class = {
    "company_fundamentals": CompanyFundamentals,
    "daily_timeseries": DailyTimeSeries,
    "balance_sheet_annual": AnnualBalanceSheet,
    "balance_sheet_quarterly": QuarterlyBalanceSheet,
    "cash_flow_annual": AnnualCashFlow,
    "cash_flow_quarterly": QuarterlyCashFlow,
    "earnings_annual": AnnualEarnings,
    "earnings_quarterly": QuarterlyEarnings,
    "income_statement_annual": AnnualIncomeStatement,
    "income_statement_quarterly": QuarterlyIncomeStatement,
    "insider_transactions": InsiderTransactions,
    "stock_splits": StockSplit,
    "dividends": Dividends,
}