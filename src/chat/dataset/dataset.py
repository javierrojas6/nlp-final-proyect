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

        df_categories_map = pd.read_csv(os.path.join(
            self.path, self.ontology_path, 'categories.csv'))

        for row in df_categories_map.iterrows():
            df.loc[df['scenario'] == row[1]['scenario'],
                   'intent'] = row[1]['intent']
            df.loc[df['scenario'] == row[1]['scenario'],
                   'success'] = row[1]['success']

        df.loc[df['scenario'] == 'auto template 11 ',
               'scenario'] = 'auto template 11'

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

    def get_chat_lines_dataframe(self, size=100):
        cds = np.array([])
        ids = np.array([])
        speakers = np.array([])
        text = np.array([])

        for conversation in self.jsonObj[:size]:
            for row in conversation['utterances']:
                cds = np.append(cds, conversation['conversation_id'])
                ids = np.append(ids, row['index'])
                speakers = np.append(speakers, row['speaker'])
                text = np.append(text, row['text'])

        columns = np.array([cds, ids, speakers, text]).T
        return pd.DataFrame(data=columns, columns=['conversation_id', 'index', 'speaker', 'text'])

    def get_clean_text(self, text, segments):
        text = text.replace('\u2019', '\'')
        real_text = ''
        last_index = 0
        for seg in segments:
            real_text += text[last_index:seg['start_index']]
            real_text += '['+seg['annotations'][0]['name']+']'
            last_index = seg['end_index']
        real_text += text[last_index:]
        return real_text

    def get_chat_lines_train_dataframe(self, size=100):
        cds = np.array([])
        user = np.array([])
        assistant = np.array([])

        for conversation in self.jsonObj[:size]:
            tmp_user = np.array([])
            tmp_assistant = np.array([])
            previous = ''

            for row in conversation['utterances']:
                real_text = row['text']

                if 'segments' in row.keys():
                    real_text = self.get_clean_text(row['text'],  row['segments'])

                if row['speaker'] == 'user':
                    tmp_user = np.append(tmp_user, real_text)

                if row['speaker'] == 'assistant':
                    tmp_assistant = np.append(tmp_assistant, real_text)

                if row['speaker'] == 'user' and previous == 'user':
                    tmp_assistant = np.append(tmp_assistant, '')

                if row['speaker'] == 'assistant' and previous == 'assistant':
                    tmp_user = np.append(tmp_user, '')

                previous = row['speaker']

            if tmp_user.shape[0] > tmp_assistant.shape[0]:
                tmp_assistant = np.append(tmp_assistant, [''] * (tmp_user.shape[0] - tmp_assistant.shape[0]))
            else:
                tmp_user = np.append(tmp_user, [''] * (tmp_assistant.shape[0] - tmp_user.shape[0]))

            user = np.append(user, tmp_user)
            assistant = np.append(assistant, tmp_assistant)
            cds = np.append(cds, np.array([conversation['conversation_id']] * tmp_user.shape[0]))

        columns = np.array([cds, user, assistant]).T
        return pd.DataFrame(data=columns, columns=['conversation_id', 'user', 'assistant'])