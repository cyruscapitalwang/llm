# src/langchain_chaining.py  (underscore in filename is safer than '-')
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env into os.environ

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = """List 5 restaurants you can find in Nile, Illinois for the specific ethnics {topic}.
List any information you find 

EXAMPLE: TOOL_NAME

YOUR RESPONSE:
"""
prompt = PromptTemplate.from_template(template)

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic": "Chinese"}))
