import os
from dotenv import load_dotenv
import pickle
import json
from typing import Any, List
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship


# load .env file
load_dotenv()

# initialize graph and llm
uri = os.environ['URI']
usr = os.environ['USERNAME']
psw = os.environ['PASSWORD_DD']
graph = Neo4jGraph(url=uri, username=usr, password=psw)


class Element(BaseModel):
    type: str
    text: Any


# processing a single JSON into a GraphDocument
def process_json_response(json_data, filename) -> GraphDocument:
    print(f"Processing file: {filename}")
    
    # extract relations from JSON
    relations = json_data.get("relations", [])

    nodes_set = set()
    relationships = []

    for rel in relations:
        try:
            # add nodes to set for deduplication
            nodes_set.add((rel["head"], rel["head_type"]))
            nodes_set.add((rel["tail"], rel["tail_type"]))
            
            # create source and target entity nodes
            source_node = Node(
                id=rel["head"],
                type=rel["head_type"]
            )
            target_node = Node(
                id=rel["tail"],
                type=rel["tail_type"]
            )
            
            # create relationship
            relationships.append(
                Relationship(
                    source=source_node,
                    target=target_node,
                    type=rel["relation"]
                )
            )

        except Exception as e:
            print(f"Error processing relation in {filename}: {rel}, Error: {e}")

    # build list of nodes
    nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]

    # create a Document using the title and abstract
    source_content = f"Title: {json_data.get('title', 'Unknown')}\nAbstract: {json_data.get('abstract', 'Unknown')}"
    source_doc = Document(
        page_content=source_content,
        metadata={"filename": filename}
    )
    
    # return constructed GraphDocument
    return GraphDocument(nodes=nodes, relationships=relationships, source=source_doc)


# process all JSONs in given folder and create Neo4j graph
def process_json_folder(folder_path: str = "platinum_relations") -> None:
    # check if folder exists
    if not os.path.exists(folder_path):
        print(f"{folder_path} does not exist")
        return
    
    # extract all JSONs
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # check for cached results
    pickle_file = "json_graph_output.pkl"
    if os.path.exists(pickle_file):
        print("Found cached results")
        with open(pickle_file, "rb") as f:
            graph_documents = pickle.load(f)
    else:
        print("Processing JSON files")
        graph_documents = []
        
        for i, filename in enumerate(json_files):
            file_path = os.path.join(folder_path, filename)
            print(f"Opening {filename}")
            
            try:
                # load JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # process data and store resulting GraphDocument
                graph_doc = process_json_response(json_data, filename)
                graph_documents.append(graph_doc)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        # cache the results for future re-use
        with open(pickle_file, "wb") as f:
            pickle.dump(graph_documents, f)

    # clear graph if already populated
    graph.query("MATCH (n) DETACH DELETE n")
    
    # add GraphDocument to Neo4j graph
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True, 
        include_source=True
    )
    
    print(f"Complete")

if __name__ == "__main__":
    process_json_folder("platinum_relations")