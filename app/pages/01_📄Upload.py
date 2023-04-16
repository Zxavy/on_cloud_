import os
import glob

import traceback
import pandas as pd
import streamlit as st
from contextlib import contextmanager


@contextmanager
def st_exceptions():
    try:
        yield
    except Exception as e:
        st.error(str(e))
        st.error(traceback.format_exc())


def load_data(file):
    with st_exceptions():
        df = pd.read_csv(file, index_col=None)
    return df

def load_data_from_folder():
    folder_path = "./artifacts/data/sample_data"
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        st.warning("No CSV files found in the specified folder.")
        return None, None
    
    dataset_options = ["None"] + [os.path.basename(f) for f in csv_files]
    dataset_choice = st.selectbox("Select a dataset from the folder", dataset_options)

    if dataset_choice != "None":
            with st.spinner("Loading dataset..."):
                df = pd.read_csv(os.path.join(folder_path, dataset_choice))
                st.session_state.df = df
                st.dataframe(df)
                if 'preprocessed_df' in st.session_state:  # Add this condition
                    st.session_state.pop('preprocessed_df', None)  # Add this line
                    st.session_state.pop('split_data_done', None)  # Add this line
                return dataset_choice, df
    else:
        st.warning("Please select a dataset from the folder.")
        return None, None




# def load_data_from_folder():
#     folder_path = "./artifacts/data/sample_data"
#     csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
#     if not csv_files:
#         st.warning("No CSV files found in the specified folder.")
#         return None
    
#     dataset_options = ["None"] + [os.path.basename(f) for f in csv_files]
#     dataset_choice = st.selectbox("Select a dataset from the folder", dataset_options)

#     if dataset_choice != "None":
#         with st.spinner("Loading dataset..."):
#             df = pd.read_csv(os.path.join(folder_path, dataset_choice))
#             st.session_state.df = df
#             st.dataframe(df)
#     else:
#         st.warning("Please select a dataset from the folder.")



st.header("Upload Your Dataset")
st.divider()
st.write("Please upload your dataset in CSV format or select a dataset from a folder located outside the Streamlit application folder. The dataset will be stored in the session state and used for further processing.")


loaded_data_name, df = load_data_from_folder()
if loaded_data_name:
    st.session_state.data_name = loaded_data_name
    st.session_state.df = df
else:
    st.session_state.data_name = "Not loaded"
    if 'df' in st.session_state:
        del st.session_state.df
st.divider()



file = st.file_uploader("Or, upload your own dataset (CSV format)")

if file:
    with st_exceptions():
        df = load_data(file)
        st.session_state.df = df
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
        st.session_state.data_name = file.name
        if 'preprocessed_df' in st.session_state:  
            st.session_state.pop('preprocessed_df', None)  
            st.session_state.pop('split_data_done', None)  
        
if "data_name" in st.session_state:
    st.markdown(f"Loaded Data: <span style='color:orange;'>{st.session_state.data_name}</span>", unsafe_allow_html=True)
else:
    st.session_state.data_name = "Not loaded"
    st.markdown(f"Loaded Data: <span style='color:orange;'>{st.session_state.data_name}</span>", unsafe_allow_html=True)
