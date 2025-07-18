import streamlit as st
import pandas as pd
import fish_pred
import pickle

DATA_URL= "Fish.csv"
if "weight" not in st.session_state:
    st.session_state.weight = 20
if "height" not in st.session_state:
    st.session_state.height = 10
if "width" not in st.session_state:
    st.session_state.width = 5
if "length1" not in st.session_state:
    st.session_state.length1 = 10
if "length2" not in st.session_state:
    st.session_state.length2 = 10
if "length3" not in st.session_state:
    st.session_state.length3 = 10
if "prediction" not in st.session_state:
    st.session_state.prediction = pd.DataFrame(columns=["Weight", "Height", "Width", "Length1", "Length2", "Length3", "Bream_Probability", "Perch_Probability"])

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    return data
st.title("Fish Prediction")
data = load_data()
species = data["Species"].unique()
with st.expander("Dataset"):
    st.write(data)
    for specie in species:
        st.write(specie + " Data")
        specie_df = data[data["Species"] == specie]
        st.write(specie_df)

with st.expander("Data Exploration"):
    st.write(data["Species"].value_counts())
    data = data[data["Species"].isin(["Perch", "Bream"])]
    st.write("New Data -- Only keeping the top species")
    st.write(data)


with st.expander("Data Visualization"): 
    species_feature_mean = data.groupby("Species").mean(numeric_only=True)
    st.write(species_feature_mean)
    species_feature_mean.drop(columns="Category", inplace = True)
    st.markdown("#### Mean value of each feature for fish species") 
    #plot the average weight, height, width of each
    for column in species_feature_mean.columns:
        st.subheader("Average " + column + " of fish species")
        st.bar_chart(species_feature_mean[[column]]) #select a column keep species at index
    st.markdown("#### Plots of feature comparisons")
    #height vs weight of each species
    st.scatter_chart(data =data, x = "Height", y = "Weight", color="Species")
    #length1 vs width of each species
    st.scatter_chart(data =data, x = "Width", y = "Length1", color="Species")

with st.sidebar:
    st.header("Input Features")
    st.slider("Weight (g)", 5.9,  1000.0, key = "weight")
    st.slider("Height (cm)", 2.0,  19.0, key = "height")
    st.slider("Width (cm)", 1.4,  8.2, key = "width")
    st.slider("Length1 (cm)", 7.5,  41.2, key = "length1")
    st.slider("Length2 (cm)", 8.4, 44.0, key = "length2")
    st.slider("Length3 (cm)", 8.8,  46.7, key = "length3")

st.subheader("Predictions")

input_data = {
    "Weight": st.session_state.weight,
    "Height": st.session_state.height,
    "Width": st.session_state.width,
    "Length1": st.session_state.length1,
    "Length2": st.session_state.length2,
    "Length3": st.session_state.length3
}

# loading the saved models
fish_model = pickle.load(open('models/fish_model.sav', 'rb'))

if st.button("predict"):
    input_df = pd.DataFrame([input_data])
    probs = fish_model.predict_proba(input_df)
    new_prediction = pd.DataFrame(probs, columns=["Bream_Probability", "Perch_Probability"])
    new_row = pd.concat([input_df, new_prediction], axis=1)
    st.session_state.prediction = pd.concat([st.session_state.prediction, new_row], ignore_index=True)
st.write(st.session_state.prediction)










