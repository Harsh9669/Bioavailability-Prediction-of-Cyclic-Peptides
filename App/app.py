import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd
import pickle

# Loading trained models and scaler
with open('App/permeability_model.pkl', 'rb') as f:
    permeability_model = pickle.load(f)

with open('App/gastric_stability_model.pkl', 'rb') as f:
    gastric_stability_model = pickle.load(f)

with open('App/intestinal_stability_model.pkl', 'rb') as f:
    intestinal_stability_model = pickle.load(f)

with open('App/bioavailability_model.pkl', 'rb') as f:
    bioavailability_model = pickle.load(f)

with open('App/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to calculate RDKit descriptors
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descriptors = {
        'TPSA': Descriptors.TPSA(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'MolWt': Descriptors.MolWt(mol)
    }

    # Convert the descriptors to a pandas DataFrame
    descriptor_df = pd.DataFrame(list(descriptors.items()), columns=['Descriptor', 'Value'])

    # Display the table in Streamlit
    st.write("Molecular Descriptors:")
    st.table(descriptor_df)

    # Return the descriptors as a numpy array
    return np.array([descriptors[desc] for desc in descriptors])


# Function to map stability predictions
def stability_mapping(value):
    if value == 0:
        return 'Unstable'
    elif value == 1:
        return 'Partially Stable'
    elif value == 2:
        return 'Stable'

# Function to calculate overall stability
def calculate_overall_stability(gastric_stability, intestinal_stability):
    if gastric_stability == 2 and intestinal_stability == 2:
        return 1 
    elif (gastric_stability == 1 and intestinal_stability >= 1) or (gastric_stability == 2 and intestinal_stability == 1):
        return 0.8  
    elif (gastric_stability == 2 and intestinal_stability == 0) or (gastric_stability == 0 and intestinal_stability == 2):
        return 0.6  
    elif (gastric_stability == 1 and intestinal_stability == 0) or (gastric_stability == 0 and intestinal_stability == 1):
        return 0.4  
    else:
        return 0  

# Streamlit UI
st.title('ORAL BIOAVALABILITY PREDICTOR')
st.subheader('Predict the expected oral bioavailability of cyclic peptides using SMILES string')

# Input SMILES string
smiles_input = st.text_input('Enter a SMILES string:', '')

if smiles_input:
    try:
        # Calculate descriptors
        descriptors = calculate_descriptors(smiles_input)
        
        # Check if descriptors are within the scaler range
        min_range = scaler.data_min_
        max_range = scaler.data_max_
        
        out_of_range = np.any((descriptors < min_range) | (descriptors > max_range))
        
        if out_of_range:
            st.warning("Warning: One or more descriptors are outside the scaler's range.")
        
        # Scale descriptors
        descriptors_scaled = scaler.transform([descriptors])
        
        # Predict permeability
        permeability = permeability_model.predict(descriptors_scaled)[0]
        permeability_rounded = round(float(permeability), 2)
        st.write(f"Predicted Permeability (LogP): {permeability_rounded}")
        
        # Predict gastric stability
        gastric_stability = gastric_stability_model.predict(descriptors_scaled)[0]
        gastric_stability_mapped = stability_mapping(gastric_stability)
        st.write(f"Predicted Gastric Stability: {gastric_stability_mapped}")
        
        # Predict intestinal stability
        intestinal_stability = intestinal_stability_model.predict(descriptors_scaled)[0]
        intestinal_stability_mapped = stability_mapping(intestinal_stability)
        st.write(f"Predicted Intestinal Stability: {intestinal_stability_mapped}")
        
        # Calculate overall stability based on given conditions
        overall_stability = calculate_overall_stability(gastric_stability, intestinal_stability)
        
        # Prepare bioavailability input
        bioavailability_input = np.append(descriptors_scaled, [permeability, overall_stability])
        
        # Predict bioavailability
        bioavailability = bioavailability_model.predict([bioavailability_input])[0]
        bioavailability_rounded = round(float(bioavailability), 2)
        st.subheader("Predicted Bioavailability:")
        st.info(f"{bioavailability_rounded} %")

    except Exception as e:
        st.error(f"Error processing SMILES: {e}")

# Contact Us section
st.sidebar.subheader("Contact Us")
st.sidebar.write("For any inquiries, please contact us at:")
st.sidebar.write("[hj728490@gmail.com](mailto:hj728490@gmail.com)")
st.sidebar.write("[Link to the article](#)")
