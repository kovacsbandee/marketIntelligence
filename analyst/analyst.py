from symbol.symbol import Symbol

class Analyst:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.symbol = None  # This can be set later if needed
        self.llm_to_be_used = "gpt-41"  # Default LLM, can be changed as needed

    def connect_to_LLM_provider(self, llm_provider: str):
        """
        This method connects the analyst to a specified LLM provider.
        In the future I'm planning to add OpenAI, and Claude as options.
        This method does the setting, load the API keys from environment variables, and tests the connection.
        The output is a self.llm attribute that is used in the analyze method.
        Capable of receiving prompts, knowledge bases, data from the alphavantage and plots from the infrastructure/ui/dash/plots directory.
        """
        # Placeholder for actual connection logic

        if self.llm_to_be_used == llm_provider:
            return True
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
    def get_alphavantage_analysis(self, symbol: str):
        """
        Retrieve analysis data from Symbol company_fundamentals.
        
        This method should be implemented by subclasses to perform specific analyses.
        
        Args:
            symbol (str): The stock symbol to analyze.
        
        Returns:
            dict: The analysis data retrieved from Alpha Vantage.
        """
        raise NotImplementedError("Subclasses should implement this method")


    def analyze(self, data):
        raise NotImplementedError("Subclasses should implement this method")

    def __repr__(self):
        return f"Analyst(name={self.name}, description={self.description})"
    

class FinancialAnalyst(Analyst):
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        # Additional attributes specific to financial analysts can be added here
    def analyze(self, data):
        """
        Perform financial analysis on the provided data.
        
        This method should be implemented by subclasses to perform specific financial analyses.
        
        Args:
            data (dict): The financial data to analyze.
        
        Returns:
            dict: The results of the analysis.
        """
        raise NotImplementedError("Financial analysis logic is not yet implemented")
    def __repr__(self):
        return f"FinancialAnalyst(name={self.name}, description={self.description})"    
        

class QuantitativeAnalyst(Analyst):
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        # Additional attributes specific to quantitative analysts can be added here
    def analyze(self, data):
        """
        Perform quantitative analysis on the provided data.
        This method should be implemented by subclasses to perform specific quantitative analyses.
        Args:
            data (dict): The quantitative data to analyze.
        Returns:
            dict: The results of the analysis.
        """
        raise NotImplementedError("Quantitative analysis logic is not yet implemented")
    def __repr__(self):
        return f"QuantitativeAnalyst(name={self.name}, description={self.description})" 
    
class NewsAnalyst(Analyst):
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        # Additional attributes specific to news analysts can be added here
    def analyze(self, data):
        """
        Perform news analysis on the provided data.
        This method should be implemented by subclasses to perform specific news analyses.
        Args:
            data (dict): The news data to analyze.
        Returns:
            dict: The results of the analysis.
        """
        raise NotImplementedError("News analysis logic is not yet implemented") 
    def __repr__(self):
        return f"NewsAnalyst(name={self.name}, description={self.description})"


class LLMAnalyst(Analyst):
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        # Additional attributes specific to LLM analysts can be added here
    def analyze(self, data):
        """
        Perform analysis using a Large Language Model (LLM) on the provided data.
        This method should be implemented by subclasses to perform specific LLM analyses.
        Args:
            data (dict): The data to analyze using LLM.
        Returns:
            dict: The results of the analysis.
        """
        raise NotImplementedError("LLM analysis logic is not yet implemented")
    def __repr__(self):
        return f"LLMAnalyst(name={self.name}, description={self.description})"
    
