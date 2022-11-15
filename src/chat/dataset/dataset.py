import json
import os

import numpy as np
import pandas as pd


class Dataset:
  jsonObj = []
  path = None
  data_path = 'data'
  ontology_path = 'ontology'

  def __init__(self, path):
    self.path = path
    files = np.array(os.listdir(os.path.join(path, self.data_path)))

    self.jsonObj = []
    for file in files:
      tmp_obj = json.load(open(os.path.join(path, self.data_path, file)))
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

    df = pd.DataFrame(data=columns, columns=[
        'conversation_id',
        'scenario',
        'instructions',
        'conversation',
        # 'utterances'
    ])

    # loads extra data
    df['intent'] = [''] * df.shape[0]
    df['success'] = [''] * df.shape[0]

    df_categories_map = pd.read_csv(os.path.join(self.path, self.ontology_path, 'categories.csv'))

    for row in df_categories_map.iterrows():
      df.loc[df['scenario'] == row[1]['scenario'], 'intent'] = row[1]['intent']
      df.loc[df['scenario'] == row[1]['scenario'], 'success'] = row[1]['success']

    df.loc[df['scenario'] == 'auto template 11 ', 'scenario'] = 'auto template 11'

    return df

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
