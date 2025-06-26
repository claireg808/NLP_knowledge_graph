import os
import pandas as pd
import json
from collections import defaultdict
from pathlib import Path


entity_counts = defaultdict(lambda: {'head_count': 0, 'tail_count': 0, 'labels': set()})
relation_counts = defaultdict(int)


# extract entity & relation data into global defaultdicts
def extract_data(path):
    with open(path, 'r') as file:
        global entity_counts
        global relation_counts

        data = json.load(file)
        relations = data.get('relations', [])

        for relation_data in relations:
            head = relation_data['head']
            tail = relation_data['tail']
            head_type = relation_data['head_type']
            tail_type = relation_data['tail_type']
            # remove _, lowercase
            relation_name = relation_data['relation'].replace("_", " ").lower()

            # add count for head/ tail entity, and add label if not added previously
            entity_counts[head]['head_count'] += 1
            if head_type not in entity_counts[head]['labels']:
                entity_counts[head]['labels'].add(head_type)
            entity_counts[tail]['tail_count'] += 1
            if head_type not in entity_counts[tail]['labels']:
                entity_counts[tail]['labels'].add(tail_type)
            
            # count relation occurrence
            relation_counts[relation_name] += 1


# find all _relations files in the given folder
def walk_directory(relations_output):
    # parent directory
    for root, _, files in os.walk(relations_output):
        # entity files
        for file in files:
            # samples
            if file.endswith('_relations.json'):
                path = Path(os.path.join(root, file))
                try:
                    extract_data(path)
                except Exception as e:
                    print(f'Exception: {str(e)}\nFile: {str(file)}')


if __name__ == '__main__':
    entity_folder_name = 'platinum_relations'
    walk_directory(entity_folder_name)

    # make entities df
    entities_data = []
    for entity_name, counts in entity_counts.items():
        entities_data.append({
            'entity_name': entity_name,
            'head_count': counts['head_count'],
            'tail_count': counts['tail_count'],
            'labels': list(counts['labels'])  # Convert set to list for DataFrame
        })

    pd.set_option('display.max_rows', None)
    entities_df = pd.DataFrame(entities_data)
    entities_df_sorted = entities_df.sort_values(by='head_count', ascending=False)
    entities_df_sorted.to_csv('entity_counts.csv', index=False)


    # make relations df
    relations_data = []
    for relation_name, count in relation_counts.items():
        relations_data.append({
            'relation_name': relation_name,
            'relation_count': count
        })

    pd.set_option('display.max_rows', None)
    relations_df = pd.DataFrame(relations_data)
    relations_df_sorted = relations_df.sort_values(by='relation_count', ascending=False)
    relations_df_sorted.to_csv('relation_counts.csv', index=False)