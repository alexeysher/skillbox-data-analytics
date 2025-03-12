import streamlit as st
from streamlit import config
from auxiliary import connect_gcs, load_data_from_gcs, css_styling
from functions import trimean_mod, trimean_mod_diff

# Google Cloud
GC_CREDENTIAL_INFO = st.secrets['gc-service-account'] # Credential info
GC_BUCKET_ID = st.secrets['gc-storage']['bucket_id'] # Bucket id
GC_BUCKET = connect_gcs(GC_CREDENTIAL_INFO, GC_BUCKET_ID) # Bucket
GC_DATA_PATH = 'data' # Data folder path


@st.cache_resource(show_spinner='Loading...')
def prepare_data():
    names = (
        'data', 'data_clean', 'metrics', 'statistic_distributions',
        'section_6_3', 'section_6_4_1', 'section_6_4_2', 'section_6_4_3',
        'section_7_3', 'section_7_4_1', 'section_7_4_2', 'section_7_4_3', 'section_7_4_4',
        'section_8_3', 'section_8_4_1', 'section_8_4_2', 'section_8_4_3', 'section_8_4_4',
        'section_9_3', 'section_9_4_1', 'section_9_4_2', 'section_9_4_3',
    )
    return {name: load_data_from_gcs(f'{name}.pkl', GC_BUCKET, GC_DATA_PATH) for name in names}


st.set_page_config(page_title='Research of MegaFon (large mobile and telecom operator) customer success survey',
                   page_icon='bar-chart', layout='wide')

css_styling()

prepare_data()

pages = [
    st.Page('sections/title.py', title="Title", default=True),
    st.Page('sections/problem_statement.py', title="Problem statement"),
    st.Page('sections/provided_data.py', title='Provided data'),
    st.Page('sections/data_cleaning.py', title="Data cleaning"),
    st.Page('sections/eda.py', title="Exploratory data analysis"),
    st.Page('sections/setting_objectives.py', title="Setting the objectives"),
    st.Page('sections/selection.py', title="Selection of metrics, statistics and criteria"),
    st.Page('sections/mobile_internet_dissatisfaction.py',
            title="Research of reasons for dissatisfaction with mobile Internet service"),
    st.Page('sections/mobile_internet_assessment.py',
            title="Research of mobile internet service quality assessments"),
    st.Page('sections/mobile_internet_csat.py',
            title="Research of satisfaction levels with mobile internet service"),
    st.Page('sections/metric_influence.py',
            title="Research of the influence of metrics on the customer satisfaction level "
                  "with Mobile Internet service"),
    st.Page('sections/summary.py', title="Summary")
]

pg = st.navigation(pages)
pg.run()

st.session_state.update(prepare_data())

st.session_state.research_metrics = st.session_state.metrics.loc[[
    'Downlink Throughput(Kbps)',
    'Video Streaming Download Throughput(Kbps)',
    'Web Page Download Throughput(Kbps)'
]]
st.session_state.research_metrics['statistic'] = trimean_mod
st.session_state.research_metrics['test statistic'] = trimean_mod_diff

st.session_state.alpha = 0.05
st.session_state.betta = 1 - st.session_state.alpha

config.dataFrameSerialization = "arrow"
