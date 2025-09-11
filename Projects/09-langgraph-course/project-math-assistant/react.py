from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()

@tool
def square(num: float) -> float:
    """
    param num: the number to square
    returns: the square of the number
    """
    return float(num) ** 2


@tool
def add(num: float) -> float:
    """
    param num: the number to add
    returns: the sum of the number
    """
    return float(num) + float(num)


tools = [square, add]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)