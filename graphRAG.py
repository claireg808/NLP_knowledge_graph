import re
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

# load .env file
load_dotenv()

# initialize graph and llm
uri = os.environ['URI']
usr = os.environ['USERNAME']
psw = os.environ['PASSWORD_DD']

graph = Neo4jGraph(url=uri, username=usr, password=psw, refresh_schema=False)

llm = ChatOpenAI(
    base_url=os.environ['BASE_URL'],
    api_key=os.environ['API_KEY'],
    model=os.environ['MODEL'],
    timeout=float(os.environ['TIMEOUT'])
)

# initialize map & reduce prompts
MAP_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
        {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
    ]
}}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Data: Reports (report ids)]"

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

Do not include information where the supporting evidence for it is not provided.


---Data tables---

{context_data}
"""

map_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            MAP_SYSTEM_PROMPT,
        ),
        (
            "human",
            "{question}",
        ),
    ]
)
map_chain = map_prompt | llm | StrOutputParser()

REDUCE_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Analyst Reports---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

reduce_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            REDUCE_SYSTEM_PROMPT,
        ),
        (
            "human",
            "{question}",
        ),
    ]
)
reduce_chain = reduce_prompt | llm | StrOutputParser()

response_type: str = "multiple paragraphs"


def global_retriever(query: str, level: int, response_type: str = response_type) -> str:
    # query graph to find all communities of the given level
    community_data = graph.query(
        """
    MATCH (c:__Community__)
    WHERE c.level = $level
    RETURN c AS output
    """,
        params={"level": level},
    )

    # append all community info together
    community_info = []
    for community in tqdm(community_data, desc="Processing communities"):
        community_info.append(community['output']['summary'] + '\n')

     # query for intermediate + final response
    intermediate_response = map_chain.invoke(
            {"question": query, "context_data": community_info}
        )
    
    final_response = reduce_chain.invoke(
        {
            "report_data": intermediate_response,
            "question": query,
            "response_type": response_type,
        }
    )

    match = re.search(r".*<\/think>\s*(.*)", final_response, re.DOTALL)
    if match:
        final_response = match.group(1)

    log = {
        'user_query': query,
        'intermediate response': intermediate_response,
        'final_response': final_response
    }

    # save to log
    with open('log.json', 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)

    return final_response

print(global_retriever("What is the relationship between depression and the gut?", 2))