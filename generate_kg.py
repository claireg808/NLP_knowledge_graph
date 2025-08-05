import os
from dotenv import load_dotenv
import pickle
import json
from collections import defaultdict
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


# processing a single JSON into a GraphDocument
def process_json_response(json_data, filename):
    # extract relations from JSON
    relations = json_data.get('relations', [])

    ### do we want to deduplicate by document or globally??
    nodes_set = set()
    relationships = []

    for rel in relations:
        try:
            # add nodes to set for deduplication
            nodes_set.add((rel['head'], rel['head_type']))
            nodes_set.add((rel['tail'], rel['tail_type']))

            # add types to list for normalizing
            varying_capitals((rel['head_type'], rel['tail_type']))

            # create source and target entity nodes
            source_node = Node(
                id=rel['head'],
                type=rel['head_type']
            )
            target_node = Node(
                id=rel['tail'],
                type=rel['tail_type']
            )
            
            # create relationship between source & target
            relationships.append(
                Relationship(
                    source=source_node,
                    target=target_node,
                    type=rel['relation']
                )
            )

        except Exception as e:
            print(f'Error processing relation: {rel}, Error: {e}')

    # build list of nodes
    nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]

    # create a Document using the title and abstract
    source_content = f"Title: {json_data.get('title', 'Unknown')}\nAbstract: {json_data.get('abstract', 'Unknown')}"
    source_doc = Document(
        page_content=source_content,
        metadata={'filename': filename}
    )
    
    # return constructed GraphDocument
    return nodes, relationships, source_doc


# map all capitalization variants using lowercase as key
different_capitals = defaultdict(set)
def varying_capitals(types):
    for type in types:
        lowercase = type.lower()
        different_capitals[lowercase].add(type)

# count number of uppercase letters
def count_uppercase(type):
    return len([letter for letter in type if letter.isupper()])

# select most capitalized version of each type
def map_max_capitalized(different_capitals):    
    mapped_types = defaultdict(str)

    for _, type_set in different_capitals.items():
        type_list = list(type_set)
        # sort by number of uppercase characters descending
        if len(type_list) > 1:
            type_list.sort(key=count_uppercase, reverse=True)
        # use lowercase type as key, and most uppercase version the value
        mapped_types[type_list[0].lower()] = (type_list[0])

    return mapped_types


# create final GraphDocument
def create_normalized_graph_documents(nodes, relationships, source_doc, mapped_types):
    for node in nodes:
        node.type = mapped_types.get(node.type.lower())
    
    for relationship in relationships:
        relationship.source.type = mapped_types.get(relationship.source.type.lower())
        relationship.target.type = mapped_types.get(relationship.target.type.lower())

    return GraphDocument(nodes=nodes, relationships=relationships, source=source_doc)


# process all JSONs in given folder and create Neo4j graph
def process_json_folder(folder_path):
    # check if folder exists
    if not os.path.exists(folder_path):
        print(f'{folder_path} does not exist')
        return
    
    # extract all JSONs
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # check for cached results
    pickle_file = 'json_graph_output.pkl'
    if os.path.exists(pickle_file):
        print('Processing cached results')
        with open(pickle_file, 'rb') as f:
            graph_documents = pickle.load(f)
    else:
        print('Processing JSON files')
        document_info = []
        
        for filename in json_files:
            file_path = os.path.join(folder_path, filename)
            
            try:
                print(f'Processing file: {filename}')

                # load JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # process data and store resulting GraphDocument
                nodes, relationships, source_doc = process_json_response(json_data, filename)
                document_info.append([nodes, relationships, source_doc])
                
            except Exception as e:
                print(f'Error processing {filename}: {e}')
                continue

    # map all differing capitalizations to the most capitalized version
    mapped_types = map_max_capitalized(different_capitals)

    # build normalized GraphDocuments
    graph_documents = []
    for document in document_info:
        graph_documents.append(
            create_normalized_graph_documents(document[0], document[1], document[2], mapped_types)
        )

    # cache the results for future re-use
    with open(pickle_file, 'wb') as f:
        pickle.dump(graph_documents, f)

    # clear graph if already populated
    graph.query("MATCH (n) DETACH DELETE n")
    
    # add GraphDocuments to Neo4j graph
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True, 
        include_source=True
    )
    
    print('\n\nComplete')

if __name__ == '__main__':
    process_json_folder('platinum_relations')