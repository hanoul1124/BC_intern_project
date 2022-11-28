from kiwipiepy import Kiwi
import pandas as pd
from collections import defaultdict
from tqdm.auto import tqdm
from apyori import apriori
from matplotlib import font_manager
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import copy
import json


kiwi = Kiwi()
kiwi.add_user_word(word='바로카드', tag='NNP', score=3.0)
kiwi.add_user_word(word='비씨카드', tag='NNP', score=4.0)
kiwi.add_user_word(word='BC카드', tag='NNP', score=4.0)
kiwi.add_user_word(word='페이북', tag='NNP', score=4.0)

with open('./config.json', 'r') as config_file:
    configs = json.load(config_file)


def get_graphs(data, save_path):
    df = pd.read_csv(data)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")

    dfg = df.groupby('keyword').count().reset_index()
    ldfg = df.groupby(['keyword', 'label']).count().reset_index()

    mdf = copy.deepcopy(df)
    mdf['date'] = mdf['date'].dt.strftime('%Y-%m')
    mdfg = mdf.groupby(['date', 'keyword', 'label']).count().reset_index()
    pmdfg = mdfg[mdfg["label"] == 'Positive']
    nmdfg = mdfg[mdfg["label"] == 'Negative']

    # Figure 1
    fig = px.bar(
        dfg,
        x='keyword',
        y='document',
        title='Total classification',
        barmode='stack'
    )
    fig.update_layout(
        title=dict(
            text='Total classification',
            font=dict(
                family="Arial",
                size=22,
                color='#000000'
            )
        )
    )
    fig.write_image(f"{save_path}fig1.png")

    # Figure 2
    fig = px.bar(
        ldfg,
        x='keyword',
        y='document',
        color='label',
        title='Total classification by label',
        labels={'y': 'count'}
    )
    fig.update_layout(
        title=dict(
            text='Total classification by label',
            font=dict(
                family="Arial",
                size=22,
                color='#000000'
            )
        )
    )
    fig.write_image(f"{save_path}fig2.png")

    # Figure3
    fig = px.bar(
        mdfg,
        x='date',
        y='document',
        color='label',
        labels={'y': 'count'}
    )
    fig.update_layout(
        title=dict(
            text="월별 문장 감성 분류 통계",
            font=dict(
                family="Arial",
                size=22,
                color='#000000'
            )
        )
    )
    fig.write_image(f"{save_path}fig3.png")

    # Figure 4
    fig = px.line(
        pmdfg,
        x='date',
        y='document',
        color='keyword',
        markers=True
    )
    fig.update_layout(
        title=dict(
            text="월별/키워드별 부정적 감성 추이",
            font=dict(
                family="Arial",
                size=22,
                color='#000000'
            )
        )
    )
    fig.write_image(f"{save_path}fig4.png")

    # Figure 5
    fig = px.line(
        pmdfg,
        x='date',
        y='document',
        color='keyword',
        markers=True
    )
    fig.update_layout(
        title=dict(
            text="월별/키워드별 부정적 감성 추이",
            font=dict(
                family="Arial",
                size=22,
                color='#000000'
            )
        )
    )
    fig.write_image(f"{save_path}fig5.png")


def keyword_analysis(data, save_path):
    def get_top_k(data, dictionary, k, title, save_path):
        sentences = data['document']
        for sent in tqdm(sentences):
            tags = kiwi.tokenize(sent)
            for t in tags:
                if t.tag in ('NNP', 'NNG'):
                    dictionary[t.form] += 1
        word_list = sorted(dictionary.items(), reverse=True, key=(lambda x: dictionary[x[0]]))
        pdf = pd.DataFrame({'word': [i[0] for i in word_list[:k]], 'count': [j[1] for j in word_list[:k]]})
        pdf.to_csv(f'{save_path}{title}_top{k}.csv')

    def association_analysis(data, min_node, title, save_path):
        sentences = data['document']
        token_listset = list()
        for sent in tqdm(sentences):
            token_listset.append(
                [token.form for token in kiwi.tokenize(sent) if token.tag in ('NNP', 'NNG')]
            )
        token_listset = [t_list for t_list in token_listset if t_list]
        results = (list(apriori(token_listset, min_support=0.01)))
        sdf = pd.DataFrame(results)
        sdf['length'] = sdf['items'].apply(lambda x: len(x))
        aso = sdf[(sdf['length'] >= min_node) & (sdf['support'] >= 0.01)].sort_values(by='support', ascending=False)
        aso.to_csv(f'{save_path}{title}_aa.csv')

        edges = aso[aso['length'] == 2]
        G = nx.Graph()
        ar = (edges['items'])
        G.add_edges_from(ar)
        pr = nx.pagerank(G)
        n_size = np.array([v for v in pr.values()])
        n_size = 4000 * (n_size - min(n_size)) / min(n_size)
        g_layout = nx.circular_layout(G)
        plt.figure(figsize=(16, 12))
        plt.axis('off')
        nx.draw_networkx(
            G,
            font_size=20,
            font_family="NanumBarunGothic",
            pos=g_layout,
            node_color=list(pr.values()),
            node_size=n_size,
            alpha=0.5,
            edge_color='.5',
            cmap=plt.cm.YlGn
        )
        plt.savefig(f'{save_path}{title}_graph.png', bbox_inches='tight')

    df = pd.read_csv(data)
    df = df.drop(columns=['Unnamed: 0'])
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")

    pos_sent = df[df['label'] == 'Positive'].reset_index()
    pos_sent = pos_sent.drop(columns=['index'])

    neu_sent = df[df['label'] == 'Neutral'].reset_index()
    neu_sent = neu_sent.drop(columns=['index'])

    neg_sent = df[df['label'] == 'Negative'].reset_index()
    neg_sent = neg_sent.drop(columns=['index'])

    pos_word_dict, neg_word_dict, neu_word_dict = defaultdict(int), defaultdict(int), defaultdict(int)
    for dictionary, title in zip(
        [pos_word_dict, neg_word_dict, neu_word_dict],
        ['pos', 'neg', 'neu']
    ):
        get_top_k(data, dictionary, 20, title, save_path)
    min_node = configs["min_association_pair"]
    association_analysis(data, min_node, 'total', save_path)


def postprocessing(file, save_path):
    data = pd.read_csv(file)
    get_graphs(data, save_path)
    keyword_analysis(data, save_path)
