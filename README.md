# Virtual Medical Assistant (VMA)

## Overview

Accessing timely and precise medical information within the current healthcare system can pose significant challenges, leading to delays in decision-making and healthcare management. Traditional methods, such as scheduling appointments or conducting online searches, often prove to be time-consuming.

The **Virtual Medical Assistant (VMA)** was developed to address these obstacles, utilizing AI technology to provide instant and accurate medical support. The VMA is designed to be a resource for guidance on home-based remedies and self-care practices, effectively eliminating the need for scheduling appointments for minor health issues such as flus and fevers.

## Key Features

- **Personalized Interactions:** Provides tailored responses to individual user needs, enhancing engagement and effectiveness.
- **Responsiveness and Adaptability:** Understands context, recalls previous conversations, and offers ongoing support for health-related inquiries, facilitating a dynamic and efficient digital assistant.

## Tools Utilized

- **OpenAI:** Utilized for API keys to power the conversational capabilities of the VMA.
- **Langchain Framework:** Incorporates various components such as:
  - **Prompts:** For structuring input to the model.
  - **Chains:** For sequencing calls.
  - **Agents:** For using an LLM (Large Language Model) to determine action sequences.
- **DuckDuckGo Search Tool:** Facilitates the construction of the conversational agent and enables the VMA to effectively address user queries.

## Working Mechanism

The VMA operates as follows:

1. **Information Retrieval:** Gathers information from the MedlinePlus website and utilizes the DuckDuckGo search tool to address user inquiries.
2. **Conversation Structuring:** Employs a custom prompt template to guide the structure of conversations, ensuring clarity and organization.
3. **Output Parsing:** Uses a custom output parser to convert language model outputs into a structured format for coherent responses.
4. **Agent Execution:** Orchestrates user inquiries and tool interactions, while the conversation buffer window memory enhances the VMA's conversational capabilities.

