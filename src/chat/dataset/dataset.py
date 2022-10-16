import os
import numpy as np
import pandas as pd
import json

class Dataset:
    jsonObj = []

    def __init__(self, path):
        files = np.array(os.listdir(path))

        self.jsonObj = []
        for file in files:
            tmp_obj = json.load(open(path + file))
            self.jsonObj += tmp_obj

    def get_by_field(self, field):
        data = []
        for record in self.jsonObj:
            tmp_record = ''

            for line in record[field]:
                tmp_record += self._preprocess_text(line)

            data += [tmp_record]

        return np.array(data)

    def get_text_utterances(self):
        data = []
        for record in self.jsonObj:
            tmp_record = ''

            for line in record['utterances']:
                tmp_record += line['text'] + "\n"

            data += [tmp_record]
            
        return np.array(data)
    
    def get_dataframe(self):
        columns = np.array([
            self.get_by_field('conversation_id'),
            self.get_by_field('scenario'),
            self.get_by_field('instructions'),
            self.get_text_utterances(),
            # dataset.get_by_field('utterances'),
        ]).T

        _df = pd.DataFrame(data=columns, columns=[
            'conversation_id',
            'scenario',
            'instructions',
            'conversation',
            # 'utterances'
        ])
        
        # loads extra data
        _df['intent'] = [''] * _df.shape[0]
        _df['success'] = [''] * _df.shape[0]
        
        _df_categories_map = pd.read_csv('TM-3-2020/ontology/categories.csv')
        
        for row in _df_categories_map.iterrows():
            _df.loc[ _df['scenario'] == row[1]['scenario'], 'intent' ] = row[1]['intent']
            _df.loc[ _df['scenario'] == row[1]['scenario'], 'success' ] = row[1]['success']
            
        _df.loc[ _df['scenario'] == 'auto template 11 ', 'scenario' ] = 'auto template 11'
        
        return _df

    def get_corpus(self):
        text_array = self.get_text_utterances()
        corpus = ''
        for row in text_array:
            corpus += row
        return corpus
    
    def _preprocess_text(self, text):
        text = text.lower()
        text = text.replace('\n', '')
        text = text.replace('\r', '')

        return text