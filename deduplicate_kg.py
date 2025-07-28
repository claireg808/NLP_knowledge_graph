import os
import re
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from graphdatascience import GraphDataScience
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import List, Optional
from retry import retry
from collections import defaultdict


# load .env file
load_dotenv()

# initialize graph and llm
uri = os.environ['URI']
usr = os.environ['USERNAME']
psw = os.environ['PASSWORD_DD']
graph = Neo4jGraph(url=uri, username=usr, password=psw)

llm = ChatOpenAI(
    base_url=os.environ['BASE_URL'],
    api_key=os.environ['API_KEY'],
    model=os.environ['MODEL'],
    temperature=float(os.environ['TEMPERATURE']),
    max_tokens=7000,
    timeout=float(os.environ['TIMEOUT'])
)


# calculate text embeddings for entity name & description
vector = Neo4jVector.from_existing_graph(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    node_label='__Entity__',
    text_node_properties=['id', 'description'],
    embedding_node_property='embedding',
    url=uri, username=usr, password=psw
)


# project graph
gds = GraphDataScience(
    uri,
    auth=(usr, psw)
)

# create knn graph named 'entities'
graph_name = 'entities'

# drop the graph if it already exists
if gds.graph.exists(graph_name).iloc[0]:
    gds.graph.drop(graph_name)

G, result = gds.graph.project(
    graph_name,                   #  graph name
    '__Entity__',                 #  node projection
    '*',                          #  relationship projection
    nodeProperties=['embedding']  #  configuration parameters
)


# construct knn graph
similarity_threshold = 0.95

gds.knn.mutate(
  G,
  nodeProperties=['embedding'],
  mutateRelationshipType= 'SIMILAR',
  mutateProperty= 'score',
  similarityCutoff=similarity_threshold
)


# find groups of connected nodes
gds.wcc.write(
    G,
    writeProperty="wcc",
    relationshipTypes=["SIMILAR"]
)


# query to find duplicate entities
word_edit_distance = 4  # max character difference

potential_duplicate_candidates = graph.query(
    """MATCH (e:`__Entity__`)
    // remove nodes with 3 or less characters
    WHERE size(e.id) > 4
    // group by connected nodes into list nodes
    WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
    // keep lists with more than one node
    WHERE count > 1
    // flatten node list
    UNWIND nodes AS node
    // keep nodes with three characters or less differene
    WITH distinct
      [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id] AS intermediate_results
    // remove single node groups
    WHERE size(intermediate_results) > 1
    // create results list of all similar id's
    WITH collect(intermediate_results) AS results
    // combine groups together if one shares elements with another group
    UNWIND range(0, size(results)-1, 1) as index
    WITH results, index, results[index] as result
    WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
            CASE WHEN index <> index2 AND
                size(apoc.coll.intersection(acc, results[index2])) > 0
                THEN apoc.coll.union(acc, results[index2])
                ELSE acc
            END
    )) as combinedResult
    // deduplicate merged groups and combine into allCombinedResults
    WITH distinct(combinedResult) as combinedResult
    WITH collect(combinedResult) as allCombinedResults
    // remove subsets of groups
    UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
    WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
    WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
        WHERE x <> combinedResultIndex
        AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
    )
    // return list of groups of potential duplicates
    RETURN combinedResult
    """, params={'distance': word_edit_distance})

# find all nodes that differ only by capitalization
capitalization_duplicates = graph.query("""
    MATCH (e:__Entity__)
    WITH toLower(e.id) AS lc, collect(e) AS nodes
    WHERE size(nodes) > 1 AND size(lc) > 3
    RETURN [n IN nodes | n.id] AS duplicate_ids
""")

# count number of uppercase letters
def count_uppercase(entity):
    return len(re.findall(r'[A-Z]', entity))

# deduplicate different capitalization
def merge_capitalization_variants(entities: List[str]) -> List[List[str]]:
    normalized_groups = defaultdict(list)
    
    # make dict with lowercase entity & list all OG forms of this entity
    for entity in entities:
        normalized = entity.lower()
        normalized_groups[normalized].append(entity)
    
    to_merge = []
    for group in normalized_groups.values():
        # find groups with differing capitalization
        if len(group) > 1:
            # sort by number of uppercase characters descending
            sorted_entities = sorted(group, key=count_uppercase, reverse=True)
            to_merge.append(sorted_entities)
    
    return to_merge

# for each group of potential duplicates
manual_cap_merged_entities = []
filtered_candidates = []
for el in potential_duplicate_candidates:
    # find entities with same spelling + different capitalization
    cap_groups = merge_capitalization_variants(el['combinedResult'])
    if cap_groups:
        manual_cap_merged_entities.extend(cap_groups)
    else:
        filtered_candidates.append(el)
# only duplicate lists not differing by capitalization
potential_duplicate_candidates = filtered_candidates

# prompt LLM to make final decision
system_prompt = """You are a data processing assistant. Your task is to identify duplicate entities in a list and decide which of them should be merged.
The entities might be slightly different in format or content, but essentially refer to the same thing. Use your analytical skills to determine duplicates.

Here are the rules for identifying duplicates:
1. Entities with minor typographical differences should be considered duplicates.
2. Entities with different formats but the same content should be considered duplicates.
3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates.
4. If it refers to different numbers, dates, or products, do not merge results
"""
user_template = """
Here is the list of entities to process:
{entities}

Please identify duplicates, merge them, and provide the merged list.
"""


# list of duplicate entities
class DuplicateEntities(BaseModel):
    entities: List[str] = Field(
        description="Entities that represent the same object or real-world entity and should be merged"
    )
# list of duplicate entity lists
class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]] = Field(
        description="Lists of entities that represent the same object or real-world entity and should be merged"
    )


# initialize llm output format and prompt
extraction_llm = llm.with_structured_output(Disambiguate)

extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            user_template,
        ),
    ]
)

extraction_chain = extraction_prompt | extraction_llm


# query llm with given list of entities
@retry(tries=3, delay=2)
def entity_resolution(entities: List[str]) -> Optional[List[str]]:
    return [
        el.entities
        for el in extraction_chain.invoke({"entities": entities}).merge_entities
    ]


# submit llm query for each list of potential duplicates
MAX_WORKERS = 10
merged_entities = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # submitting all tasks and creating a list of future objects
    futures = [
        executor.submit(entity_resolution, el['combinedResult'])
        for el in potential_duplicate_candidates
    ]

    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Processing documents"
    ):
        to_merge = future.result()
        if to_merge:
            merged_entities.extend(to_merge)

print(merged_entities[:10])
print(manual_cap_merged_entities)

all_merged_entities = merged_entities + manual_cap_merged_entities

# write the results back to the database
graph.query("""
UNWIND $data AS candidates
CALL {
  WITH candidates
  MATCH (e:__Entity__) 
  WHERE e.id IN candidates
  WITH candidates, collect(e) AS nodes
  WITH nodes, head([n IN nodes WHERE n.id = candidates[0]]) AS survivor
  WHERE size(nodes) > 1 AND survivor IS NOT NULL
  CALL apoc.refactor.mergeNodes(nodes, {
    properties: {`.*`: 'discard'},
    mergeRels: true,
    into: survivor
  }) YIELD node
  RETURN node
}
RETURN count(*)

""", params={"data": all_merged_entities})

G.drop()

print('\n\nComplete')