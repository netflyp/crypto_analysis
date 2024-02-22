import os
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import GoogleFinanceAPIWrapper
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from crewai import Agent, Task, Crew, Process
from textwrap import dedent
import gradio as gr

class CryptoAnalysisCrew:
    def __init__(self, cryptocurrency, openai_api_key, serp_api_key):
        self.cryptocurrency = cryptocurrency
        self.openai_api_key = openai_api_key
        self.serp_api_key = serp_api_key

    def setup_agents_and_tasks(self):
        # Configure the LangChain LLM with the OpenAI API key
        llm = ChatOpenAI(
            name="gpt-4-turbo-preview",
            model="gpt-4-turbo-preview",
            temperature=0,
            max_tokens=2000,
            api_key=self.openai_api_key
        )

        # Initialize tools with the SERP API key
        google_finance_search = GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper(serp_api_key=self.serp_api_key), max_results=5)
        duckduckgo_search = DuckDuckGoSearchRun()
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

        market_analyst = Agent(
          role='Crypto Market Analyst',
          goal="""Provide cutting-edge analysis of the cryptocurrency market trends""",
          backstory="""An expert with deep insights into the volatile crypto market, leveraging technical analysis and market sentiment to predict future movements.""",
          verbose=True,
          llm=llm,
          tools=[duckduckgo_search, google_finance_search, wikipedia]  # Add the 'tools' key here
        )

        technology_analyst = Agent(
            role='Blockchain Technology Analyst',
            goal="""Deliver in-depth analysis of blockchain technologies behind cryptocurrencies""",
            backstory="""Armed with a computer science background and a passion for blockchain technology, you dissect the technical aspects and potential of various cryptocurrencies.""",
            verbose=True,
            llm=llm,
            tools=[duckduckgo_search, wikipedia]
        )

        regulatory_analyst = Agent(
            role='Regulatory Compliance Analyst',
            goal="""Keep abreast of and analyze the global cryptocurrency regulations""",
            backstory="""As someone with a keen interest in finance and law, you stay updated on the ever-evolving regulatory landscape of the crypto world.""",
            verbose=True,
            llm=llm,
            tools=[duckduckgo_search, google_finance_search, wikipedia]
        )

        investment_advisor = Agent(
            role='Crypto Investment Strategist',
            goal="""Craft strategic investment advice based on comprehensive crypto market analyses""",
            backstory="""As a seasoned investor in cryptocurrencies, you blend various analytical insights to formulate robust investment strategies.""",
            verbose=True,
            llm=llm
        )

        risk_analyst = Agent(
            role='Crypto Risk Analyst',
            goal="""Conduct a thorough risk assessment of cryptocurrencies""",
            backstory="""An expert in financial risk assessment with a focus on the volatile crypto market, using statistical tools and historical data to evaluate the risk associated with different cryptocurrencies.""",
            verbose=True,
            llm=llm,
            tools=[duckduckgo_search, google_finance_search, wikipedia]
        )

        price_movement_analyst = Agent(
            role='Crypto Price Movement Analyst',
            goal="""Analyze and predict future price movements of cryptocurrencies using technical analysis""",
            backstory="""Specializing in technical analysis, you use chart patterns, trading volumes, and historical price data to forecast the future price movements of cryptocurrencies.""",
            verbose=True,
            llm=llm,
            tools=[duckduckgo_search, google_finance_search, wikipedia]
        )

        market_trend_analysis = Task(description=dedent(f"""
            Analyze current trends, sentiment from social media, news, and technical indicators for cryptocurrency.
            Focus on identifying key support and resistance levels, trend patterns, and potential breakout or breakdown points. Use the necessary tools to obtain the required information.
            Your final report MUST include a comprehensive analysis of the current market trend, sentiment shifts, and potential impacts on the cryptocurrency. You MUST provide very SPECIFIC details.
            Make sure to use the most recent data as possible. Make sure to use the appropriate tools that you have or discuss with other agents when required to obtain the best results.
            
          """),
          agent=market_analyst,
          tools=[duckduckgo_search, google_finance_search, wikipedia]
        )

        technology_analysis = Task(description=dedent(f"""
            Evaluate the technological aspects and development progress of cryptocurrency.
            Analyze aspects like blockchain efficiency, smart contract functionality, and any recent technological updates.
            Make sure to use the appropriate tools that you have or discuss with other agents when required to obtain the best results.
            Your final report MUST include an assessment of the cryptocurrency's technological standing, its advancements, and potential future developments. You MUST provide very SPECIFIC details.
            Make sure to use the most recent data possible.
          """),
          agent=technology_analyst,
          tools=[duckduckgo_search, wikipedia]
        )

        regulatory_analysis = Task(description=dedent(f"""
            Monitor and analyze regulatory changes and announcements in te region impacting cryptocurrencies.
            Assess how these regulatory shifts could affect the market and specific cryptocurrencies. 
            Make sure to use the appropriate tools that you have or discuss with other agents when required to obtain the best results.
            Your final report must highlight the significant regulatory changes, their potential impact on the market, and any specific cryptocurrencies that might be affected. You MUST provide very SPECIFIC details.
            
          """),
          agent=regulatory_analyst,
          tools=[duckduckgo_search, google_finance_search, wikipedia]
        )

        risk_evaluation = Task(description=dedent(f"""
            Perform a comprehensive risk analysis for the cryptocurrency. Assess factors like market volatility, liquidity, historical performance, and exposure to regulatory changes. Use historical data and statistical tools to evaluate the risk.
            Your final report MUST include an assessment of the overall risk profile, potential high-risk scenarios, and recommended risk mitigation strategies for cryptocurrency. You MUST provide very SPECIFIC details.
            Make sure to use the most recent data possible.
        """),
        agent=risk_analyst,
        tools=[duckduckgo_search, google_finance_search, wikipedia]
        )
    
        price_prediction = Task(description=dedent(f"""
            Use technical analysis to predict future price movements of cryptocurrency. Focus on identifying key support and resistance levels and projected short term and long term price predictions.
            Your final report MUST include specific price projections, support and resistance levels and price movement predictions.
            Make sure to use the most recent data possible.
            
        """),
        agent=price_movement_analyst,
        tools=[duckduckgo_search, google_finance_search, wikipedia]
        )
    
        investment_recommendation = Task(description=dedent(f"""
            Review and synthesize the analyses provided by the Crypto Market Analyst, Blockchain Technology Analyst, Regulatory Compliance Analyst, Risk Analyst and Price Movement Analyst.
                                       
            Formulate a comprehensive investment strategy based on these insights.
            Your final report MUST include a detailed investment recommendation for cryptocurrencies, considering market trends, technological developments, regulatory changes, risk analysis and price movement analysis. You MUST provide very SPECIFIC details.
            Make sure to include potential risks and opportunities, and tailor the advice to the customer's specific needs. You MUST provide very SPECIFIC details.
          """),
          agent=investment_advisor

        )

                
        self.crew = Crew(
            agents=[
                market_analyst,
                risk_analyst,
                technology_analyst,
                regulatory_analyst,
                price_movement_analyst,
                investment_advisor
                                
            ],
            tasks=[
                market_trend_analysis,
                technology_analysis,
                regulatory_analysis,
                risk_evaluation,
                price_prediction,
                investment_recommendation
                
            ],
            verbose=True,
            Process=Process.sequential
        )

    def run(self):
        # Setup agents and tasks with the provided API keys
        self.setup_agents_and_tasks()

        # Execute the crew's analysis and return the result
        result = self.crew.kickoff()
        return result
        
# Gradio UI setup
def run_crypto_analysis(cryptocurrency, openai_api_key, serp_api_key):
    crypto_crew = CryptoAnalysisCrew(cryptocurrency, openai_api_key, serp_api_key)
    result = crypto_crew.run()
    return result

iface = gr.Interface(
    fn=run_crypto_analysis,
    inputs=[
        gr.Textbox(label="Cryptocurrency"),
        gr.Textbox(label="OpenAI API Key", type="password"),
        gr.Textbox(label="SERP API Key", type="password"),
    ],
    outputs="text",
    description="Enter the cryptocurrency you want to analyze along with your OpenAI and SERP API keys."
)

if __name__ == "__main__":
    iface.launch()
  
