from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel, Field
from crewai_tools import SerperDevTool
from stockpicker.tools.push_tool import PushNotificationTool

from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage


class TrendingComoany(BaseModel):
    '''A trending company in the news'''
    name: str = Field(description="The name of the company")
    ticker: str = Field(description="The ticker symbol of the company")
    reason: str = Field(description="The reason why the company is trending")


class TrendingCompanyList(BaseModel):
    '''A list of trending companies in the news'''
    companies: List[TrendingComoany] = Field(description="The list of trending companies")

class TrendingCompanyResearch(BaseModel):
    '''A report on a trending company'''
    name: str = Field(description="The name of the company")
    market_position: str = Field(description="Current market position and competitive analysis")
    future_outlook: str = Field(description="Future outlook and growth prospects")
    investment_potential: str = Field(description="Investment potential and sustainability for investment")

class TrendingCompanyResearchList(BaseModel):
    '''A list of reports on trending companies'''
    research_list: List[TrendingCompanyResearch] = Field(description="Comprehensive research on all trending companies")



@CrewBase
class Stockpicker():
    """Stockpicker crew"""

    agents = []
    tasks = []

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml' 

    @agent
    def trending_company_finder(self) -> Agent:
        return Agent(
            config=self.agents_config['trending_company_finder'],
            tools = [SerperDevTool()],
            memory=True,
        )

    @agent
    def financial_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['financial_researcher'],
            tools = [SerperDevTool()],
        )

    @agent
    def stock_picker(self) -> Agent:
        return Agent(
            config=self.agents_config['stock_picker'],
            tools = [PushNotificationTool()],
            memory=True,
        )

    @task
    def find_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config['find_trending_companies'],
            output_pydantic=TrendingCompanyList,
        )

    @task
    def research_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config['research_trending_companies'],
            output_pydantic=TrendingCompanyResearchList,
        )

    @task
    def pick_best_company(self) -> Task:
        return Task(
            config=self.tasks_config['pick_best_company'],
            output_pydantic=TrendingCompanyResearch,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Stockpicker crew"""

        manager = Agent(
            config=self.agents_config['manager'],
            allow_delegation=True,
        )

        short_term_memory = ShortTermMemory(
            storage = RAGStorage(
                embedder_config = {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small"
                    },
                },
                type="short_term",
                path="./memory"
            )
        )

        long_term_memory = LongTermMemory(
            storage = LTMSQLiteStorage(
                db_path = "./memory/long_term_memory_storage.db"
            )
        )

        entity_memory = EntityMemory(
            storage = RAGStorage(
                embedder_config = {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small"
                    }
                },
                type = "short_term",
                path = "./memory"
            )
        )



        return Crew(
            agents=self.agents,
            tasks=self.tasks, 
            process=Process.hierarchical,
            verbose=True,
            manager_agent=manager,
            memory=True,
            long_term_memory=long_term_memory,
            short_term_memory=short_term_memory,
            entity_memory=entity_memory,
        )
