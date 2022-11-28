import os.path

import pandas as pd
import torch
from transformers import AutoTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from kiwipiepy import Kiwi
import json


class DocumentFilter:
    def __init__(self, pretrained_model, add_keywords=None):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, return_attention_mask=True)
        self.bert = BertModel.from_pretrained(pretrained_model, return_dict=False)
        if add_keywords:
            self.tokenizer.add_tokens(add_keywords)
            self.bert.resize_token_embeddings(len(self.tokenizer))
        self.analyzer = Kiwi()

    def get_sentence_embedding(self, text):
        with torch.no_grad():
            features = self.tokenizer(
                text,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors='pt'
            )
            input_ids = features['input_ids']
            attention_mask = features['attention_mask']
            token_type_ids = features['token_type_ids']
            v = self.bert(input_ids, attention_mask, token_type_ids)[0]
            return torch.mean(v[0], dim=0, keepdims=True)

    def get_morpheme_tokens(self, sentence):
        morphemes = [
            m.form for m in self.analyzer.tokenize(sentence) if m.tag in ('NNP', 'NNG')
        ]
        return morphemes

    def get_cosine_similarity(self, keyword, sentence):
        morphemes = self.get_morpheme_tokens(sentence)
        n_sentence = ''.join(morphemes)

        if n_sentence:
            sim = cosine_similarity(
                self.get_sentence_embedding(keyword),
                self.get_sentence_embedding(n_sentence).item()
            )
            return sim
        return None


def base_preprocess(file, seperator, save_path):
    df = pd.read_csv(file, sep=seperator)
    if 'id' in df.columns:
        print('Drop ID column')
        df.drop(labels=['id'], axis=1, inplace=True)

    # Delete Null value rows
    print(f'Dataset has {len(df)} lines')
    print(f'Dataset Null value check: {df.isnull().sum()}')
    df.dropna(inplace=True)
    print(f'Dataset has {len(df)} lines now ...')

    # Delete Duplicates
    print(f'Delete Duplicate data in Document Column')
    print(len(df))
    df.drop_duplicates(subset=['document'], inplace=True, ignore_index=True)
    print(len(df))

    # Save CSV file
    df.to_csv(save_path, sep='\t', index=False)
    return df


def preprocessing(file, new_path):
    with open('./config.json', 'r') as config_file:
        configs = json.load(config_file)

    model = configs["embedding_model"]
    add_keywords = configs["add_keywords"]
    sim_keywords = configs["sim_keywords"]
    path = configs["dataset_dir"]

    df = base_preprocess(
        file=file,
        seperator='\t',
        save_path=path + 'raw_' + os.path.basename(os.path.normpath(file))
    )

    docufilter = DocumentFilter(model, add_keywords)
    sim_results = list()
    for sent in df['document']:
        for keyword in sim_keywords:
            sim = docufilter.get_cosine_similarity(keyword=keyword, sentence=sent)
            if sim > 0.4:
                sim_results.append(True)
                break
            else:
                sim_results.append(False)
    df.assign(sim=sim_results)
    res_df = df.loc[(df['sim'] == True)]
    res_df.to_csv(new_path, index=False)




