import sys
sys.path.append("C:\\Users\\Saad\\Desktop\\automl\\source")

import time
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from components.raw_preprocessing import DataPreprocessor

st.set_option('deprecation.showPyplotGlobalUse', False)

st.header("Data Preprocessing")
st.divider()
st.write("In this section, you can preprocess your data. After preprocessing, a series of visualizations and statistics will be generated for the preprocessed data.")

if "data_name" in st.session_state:
    st.markdown(f"Loaded Data: <span style='color:orange;'>{st.session_state.data_name}</span>", unsafe_allow_html=True)
else:
    st.session_state.data_name = "Not loaded"
    st.markdown(f"Loaded Data: <span style='color:orange;'>{st.session_state.data_name}</span>", unsafe_allow_html=True)


st.divider()
if 'df' in st.session_state:
    df = st.session_state.df
    preprocessor = DataPreprocessor(df)

    if 'preprocessed_df' in st.session_state:
        preprocessed_df = st.session_state.preprocessed_df
        st.markdown(f"Preprocessing: <span style='color:green;'>&#x2714;</span>", unsafe_allow_html=True)
        st.write("Preprocessed data:")
        st.dataframe(preprocessed_df)
        if st.button("Re-preprocess Data", disabled=True):
            st.session_state.pop('preprocessed_df', None)
            st.session_state.pop('split_data_done', None)
    else:
        if st.button("Start Preprocessing"):
            start_time = time.time()
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1, text="Preprocessing Dataset")
            preprocessed_df = preprocessor.preprocess_data()
            st.write("Preprocessed data:")
            st.dataframe(preprocessed_df)
            st.session_state.preprocessed_df = preprocessed_df
            st.success(f"Preprocessed data saved successfully. Time taken: {time.time() - start_time:.2f} seconds")

            st.markdown(f"Preprocessing: <span style='color:green;'>&#x2714;</span>", unsafe_allow_html=True)

            # Visualizations and statistics
            # st.subheader("Visualizations and Statistics")
            # with st.spinner("Generating Statistics..."):
            #     with st.expander("Correlation Heatmap"):
            #         corr = preprocessed_df.corr()
            #         f, ax = plt.subplots(figsize=(11, 9))
            #         cmap = sns.diverging_palette(220, 10, as_cmap=True)
            #         sns.heatmap(corr, cmap=cmap, center=0,
            #         square=True, linewidths=.5, cbar_kws={"shrink": .5})
            #         fig = plt.gcf()
            #         st.pyplot(fig)
            #         plt.clf()

            #     with st.expander("Distribution of RAM values"):
            #         sns.histplot(preprocessed_df['RAM'], kde=True)
            #         fig = plt.gcf()
            #         st.pyplot(fig)
            #         plt.clf()

            #     with st.expander("Lag features statistics"):
            #         lag_features = [col for col in preprocessed_df.columns if 'lag' in col]
            #         lag_stats = preprocessed_df[lag_features].describe().transpose()
            #         st.dataframe(lag_stats)
                
            #     with st.expander("Pairplot"):
            #         sns.pairplot(preprocessed_df)
            #         fig = plt.gcf()
            #         st.pyplot(fig)
            #         plt.clf()
else:
    st.warning("Please upload a dataset first.")