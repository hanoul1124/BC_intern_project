import streamlit as st
import pandas as pd
import plotly.express as px
import copy


# Base Dataset
st.header("Classified Sentences Dataframe")
base_df = pd.read_csv('./classified_sentences.csv')
base_df['date'] = pd.to_datetime(base_df['date'], format="%Y-%m-%d")
st.dataframe(base_df)

dfg = base_df.groupby('keyword').count().reset_index()
ldfg = base_df.groupby(['keyword', 'label']).count().reset_index()
mdf = copy.deepcopy(base_df)
mdf['date'] = mdf['date'].dt.strftime('%Y-%m')
mdfg = mdf.groupby(['date', 'keyword', 'label']).count().reset_index()
pmdfg = mdfg[mdfg["label"] == 'Positive']
nmdfg = mdfg[mdfg["label"] == 'Negative']

# sentence count grouped by keyword, label
st.header("키워드별 문장 총량 및 분류 결과 통계")
ldfg_fig = px.bar(
        ldfg,
        x='keyword',
        y='document',
        color='label',
        labels={'y': 'count'}
    )
st.plotly_chart(ldfg_fig)


# sentence count grouped by keyword, label, month
st.header("월별 문장 감성 분류 통계")
mdfg_fig = px.bar(
        mdfg,
        x='date',
        y='document',
        color='label',
        labels={'y': 'count'}
    )
st.plotly_chart(mdfg_fig)

# Total sentence count grouped by keyword
st.header("키워드별 문장 긍정 분류 통계 월별 추이")
pmdfg_fig = px.line(
        pmdfg,
        x='date',
        y='document',
        color='keyword',
        markers=True
    )
st.plotly_chart(pmdfg_fig)

# Total sentence count grouped by keyword
st.header("키워드별 문장 부정 분류 통계 월별 추이")
nmdfg_fig = px.line(
        nmdfg,
        x='date',
        y='document',
        color='keyword',
        markers=True
    )
st.plotly_chart(nmdfg_fig)



