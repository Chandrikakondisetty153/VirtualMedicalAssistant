#!/usr/bin/env python
# coding: utf-8

# <h2>Virtual medical assistant with memory</h2>
# 
# > A Virtual Medical Assistant is an AI-powered digital assistant designed to assist users with medical inquiries, provide information on health conditions, and offer guidance on symptoms, treatments, and more. The VMA can interact with users through natural language processing, making it user-friendly and accessible.

# In[1]:


get_ipython().system('pip install OpenAI')


# In[2]:


# Install required packages
get_ipython().system('pip -q install langchain tiktoken duckduckgo-search')


# In[3]:


get_ipython().system('pip show langchain')


# ### Getting necessary libraries

# In[4]:


from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate

from langchain import OpenAI, LLMChain
from langchain.tools import DuckDuckGoSearchRun

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import os
import langchain
from langchain.memory import ConversationBufferWindowMemory


# In[5]:


# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"


# #### Standard Implemetation of DuckDuckGo

# In[6]:


# Define tools
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for answering questions about current events."
    )
]


# ### The provided code defines a tool for searching medical information specifically on the "medlineplus.gov" website using DuckDuckGo

# In[7]:


# Function to wrap DuckDuckGo search
def duck_wrapper(input_text):
    search_results = search.run(f"site:medlineplus.gov {input_text}")
    return search_results

tools_medical = [
    Tool(
        name="Search medline plus",
        func=duck_wrapper,
        description="Useful for answering medical and pharmacological questions."
    )
]


# ### This code defines a template to guide the structure of a conversation in a healthcare advisory context. It emphasizes a systematic approach to answering user questions, with placeholders indicating where specific information should be inserted during the interaction.

# In[8]:


# Define prompt template
template = """As a knowledgeable health advisor, respond empathetically to the user's health concerns. Utilize the provided tools to offer insightful guidance. The available tools are:

{tools}

Use the following structure:

Question: the input question you must answer
Thought: always consider what to do
Action: choose from [{tool_names}]
Action Input: input for the action
Observation: result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to answer as a compassionate medical professional when giving your final answer.

Question: {input}
{agent_scratchpad}"""


# ### This class is used for formatting prompts, incorporating a structured approach involving intermediate steps, thoughts, actions, and observations.

# In[9]:


# Custom prompt template class
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


# In[10]:



# Instantiate the custom prompt template
prompt = CustomPromptTemplate(template=template, tools=tools, input_variables=["input", "intermediate_steps"])


# ### This class is used for parsing the output of a language model (LLM) and converting it into a structured format that can be used by an agent in a conversational or interactive system. 

# In[11]:


# Custom output parser class
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(return_values={"output": llm_output.split("Final Answer:")[-1].strip()}, log=llm_output)

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


# In[12]:


# Instantiate the custom output parser
output_parser = CustomOutputParser()


# ### Combining everything to set up our agent

# In[ ]:


# Instantiate OpenAI language model
llm = OpenAI(temperature=0)


# In[14]:


# Instantiate LLM chain with the custom prompt template
llm_chain = LLMChain(llm=llm, prompt=prompt)


# In[ ]:





# In[15]:


# Get tool names
tool_names = [tool.name for tool in tools]


# In[16]:


# Instantiate the agent
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)


# ### Agent Executors take an agent and tools and use the agent to decide which tools to call and in what order.Below code is setting up an AgentExecutor instance with a specific agent and a set of tools, possibly for handling interactions or responses within a conversational or interactive environment. 

# In[17]:


# Instantiate agent executor without memory
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


# ### Adding a conversational Memory

# ### The prompt below is similar to the previous one but the olny difference is that we are now passing in the previous conversations and  those are taken into account when formulating responses.

# In[18]:


# Set up the prompt template with history
template_with_history = """Respond empathetically to health concerns, utilizing the provided tools. The available tools are:

{tools}

Use the following structure:

User Concern: the user's health inquiry
Empathetic Response: a compassionate response to reassure the user
Insight: provide insightful information or considerations
Recommended Action: suggest a suitable action, choosing from [{tool_names}]
Action Input: specify the input for the recommended action
Observation: describe the anticipated result
... (repeat N times)
Thought: express understanding and empathy
Final Answer: offer a final health recommendation
Remember to speak as a compassionate medical professional and, for serious conditions, advise consulting a doctor.

Previous Conversation History:
{history}

New Question: {input}
{agent_scratchpad}"""


# In[19]:


# Instantiate the custom prompt template with history
prompt_with_history = CustomPromptTemplate(template=template_with_history, tools=tools, input_variables=["input", "intermediate_steps", "history"])


# In[20]:


# Instantiate LLM chain with history
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)


# In[21]:


# Get tool names
tool_names = [tool.name for tool in tools]


# ### Setting up the agent executor with memory

# In[22]:


# Instantiate the agent with memory
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)


# In[23]:


# Instantiate conversation buffer window memory
memory = ConversationBufferWindowMemory(k=10)


# We used Conversion buffer memory. This memory allows for storing messages and then extracts the messages in a variable. We kept the messages till a loop of 10.

# In[24]:


# Instantiate agent executor with memory
agent_executor_mem = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break


    if "search" not in agent_executor.agent.llm_chain.prompt.template.lower():
        history = []  # Provide an empty history for the initial question
        agent_executor.run(input=user_input, history=history)
    else:
        #history = [] if not use_memory else agent_executor_mem.get_memory()
        conversation_history = agent_executor_mem.get_agent_memory()
        history.extend(conversation_history)
        agent_executor_mem.run(input=user_input, use_memory=True, history= history)

    use_memory = "search" in agent_executor_mem.agent.llm_chain.prompt.template.lower()
    memory = ConversationBufferWindowMemory(k=10) if use_memory else None

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
    )


# ### Case 1
# User Input: what are the symptoms for migrane
# 
# > Entering new AgentExecutor chain...
# Thought: I should look up the symptoms of migraine
# Action: Search
# Action Input: symptoms of migraine
# 
# The user queries about the symptoms of migraines. in this case initial template without memory is used as it is the start of the conversation
# The agent decides to perform a search to gather information.
# 
# >Observation: Symptoms Migraines, which affect children and teenagers as well as adults, can progress through four stages: prodrome, aura, attack, and post-drome... Migraine without aura is the most common type.
# I now know the final answer
# 
# The agent provides detailed information about migraine symptoms, including stages and types.
# The final answer summarizes that migraine is a neurological disorder causing painful headache attacks with additional symptoms such as sensitivity to light, sound, smell, or touch, nausea, light sensitivity, and temporary visual changes.
# 
# > Finished chain.
# 
# Marks the end of the interaction.

# ### Case 2
# User Input: what are the symptoms for covid 19
# 
# > Entering new AgentExecutor chain...
# Empathetic Response: I understand your concern. It's important to stay informed about the latest health information.
# Insight: Covid-19 is a contagious respiratory illness caused by the novel coronavirus.
# Recommended Action: Search
# Action Input: Symptoms of Covid-19
# 
# The user inquires about the symptoms of COVID-19. From this case template with memory is used as already a search has been performed
# The agent responds empathetically, provides insight, and decides to perform a search.
# 
# >Observation: The most common symptoms of COVID-19 include: Fever or chills, A dry cough, and shortness of breath... COVID-19 may attack more than your lungs and respiratory system. Other parts of your body may also be affected by the disease...
# It's understandable to be worried about your health and the health of those around you.
# 
# The agent lists common symptoms of COVID-19 and acknowledges the user's concern.
# 
# >Final Answer: It's important to be aware of the symptoms of Covid-19 and to take the necessary precautions to protect yourself and others. If you experience any of the symptoms mentioned above, it's best to consult a doctor for further advice.
# 
# The final answer emphasizes the importance of being aware of COVID-19 symptoms and consulting a doctor if needed.
# 
# > Finished chain.
# 
# Marks the end of the interaction.

# ### Case 3
# User Input: home remedies for flu
# 
# > Entering new AgentExecutor chain...
# Empathetic Response: I understand that you're looking for home remedies for the flu.
# Insight: The flu is a virus that can cause serious complications, so it's important to take the necessary precautions.
# Recommended Action: Search
# Action Input: "home remedies for flu"
# 
# The user seeks home remedies for the flu.
# The agent responds empathetically, provides insight, and decides to perform a search.
# 
# >Observation: Learn how to use natural and home remedies to relieve flu symptoms, such as fever, tiredness, or chills... Find out the benefits, risks, and sources of these natural products...
# It sounds like you're looking for ways to manage your flu symptoms and boost your immune system.
# 
# The agent lists natural remedies for managing flu symptoms and responds empathetically.
# 
# >Final Answer: I recommend searching for natural and home remedies to relieve flu symptoms, such as fever, tiredness, or chills. Additionally, make sure to drink plenty of fluids and get plenty of rest. If your symptoms worsen or persist, please consult a doctor.
# 
# > Finished chain.
# 
# The final answer suggests searching for natural remedies, staying hydrated, and getting rest. It advises consulting a doctor if symptoms worsen.

# ### Case 4 
# User Input: what are the symptoms for covid 19 (again)
# 
# > Entering new AgentExecutor chain...
# Empathetic Response: I understand your concern. It's important to stay informed about the latest health information.
# Insight: Covid-19 is a contagious respiratory illness caused by the novel coronavirus.
# Recommended Action: Search
# Action Input: Symptoms of Covid-19
# 
# The user repeats the query about COVID-19 symptoms.
# The agent responds and searches for the already given response from the memory and fetches it.
# 
# >Observation: The symptoms of COVID-19 this fall are still primarily a fever, cough, and sore throat, body aches, and/or a headache... COVID infections tend to be mild in vaccinated adults, and many can care for themselves at home...
# It sounds like you're doing your best to stay informed and take care of yourself.
# 
# 
# >Final Answer: It's important to be aware of the symptoms of Covid-19, which include fever, cough, sore throat, body aches, and/or a headache. If you experience any of these symptoms, it's important to contact your doctor for further advice.
# 
# > Finished chain.
# 
# 
# User Input: quit
# 
# >Goodbye!
# 
# The user decides to end the interaction.
# The agent bids farewell.
