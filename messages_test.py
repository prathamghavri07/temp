from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()
llm=ChatAnthropic(model='claude-3-5-sonnet-20241022',temperature=0,max_tokens_to_sample=100)

messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about Langgraph')

]

result=llm.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)