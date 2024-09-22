# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the model
filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Load the data
df = pd.read_csv("Clustered_Customer_Data.csv")

# Set the background color and title
st.markdown('<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)
st.title("Market Segmentation")

# Define the input features exactly as they were during training
feature_names = [
    'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
    'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
    'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
    'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
    'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',
    'TENURE'
]

# Create the form for user input
with st.form("my_form"):
    balance = st.number_input(label='Balance', step=0.001, format="%.6f")
    balance_frequency = st.number_input(label='Balance Frequency', step=0.001, format="%.6f")
    purchases = st.number_input(label='Purchases', step=0.01, format="%.2f")
    oneoff_purchases = st.number_input(label='One-Off Purchases', step=0.01, format="%.2f")
    installments_purchases = st.number_input(label='Installments Purchases', step=0.01, format="%.2f")
    cash_advance = st.number_input(label='Cash Advance', step=0.01, format="%.6f")
    purchases_frequency = st.number_input(label='Purchases Frequency', step=0.01, format="%.6f")
    oneoff_purchases_frequency = st.number_input(label='One-Off Purchases Frequency', step=0.1, format="%.6f")
    purchases_installment_frequency = st.number_input(label='Purchases Installments Frequency', step=0.1, format="%.6f")
    cash_advance_frequency = st.number_input(label='Cash Advance Frequency', step=0.1, format="%.6f")
    cash_advance_trx = st.number_input(label='Cash Advance Trx', step=1)
    purchases_trx = st.number_input(label='Purchases TRX', step=1)
    credit_limit = st.number_input(label='Credit Limit', step=0.1, format="%.1f")
    payments = st.number_input(label='Payments', step=0.01, format="%.6f")
    minimum_payments = st.number_input(label='Minimum Payments', step=0.01, format="%.6f")
    prc_full_payment = st.number_input(label='PRC Full Payment', step=0.01, format="%.6f")
    tenure = st.number_input(label='Tenure', step=1)

    # Store user input in a list and convert to DataFrame to match feature names
    data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance,
             purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency,
             cash_advance_frequency, cash_advance_trx, purchases_trx, credit_limit, payments,
             minimum_payments, prc_full_payment, tenure]]
    data_df = pd.DataFrame(data, columns=feature_names)  # Ensure feature names are passed correctly

    submitted = st.form_submit_button("Submit")

if submitted:
    # Predict the cluster for the input data
    clust = loaded_model.predict(data_df)[0]
    st.write('Data Belongs to Cluster:', clust)

    # Filter the data for the predicted cluster
    cluster_df1 = df[df['Cluster'] == clust]
    st.write("Cluster Data:", cluster_df1)  # Show the data for the selected cluster

    # Only plot if there is data in the selected cluster
    if not cluster_df1.empty:  
        plt.rcParams["figure.figsize"] = (20, 3)  # Set figure size for all plots

        # Iterate over columns except 'Cluster'
        for c in cluster_df1.drop(['Cluster'], axis=1).columns:
            fig, ax = plt.subplots()
            
            # Plot histogram for each feature in the cluster
            sns.histplot(cluster_df1[c], ax=ax, kde=True)  # Added KDE (kernel density estimate) for better visualization
            
            st.pyplot(fig)  # Display the plot in Streamlit
            plt.close(fig)  # Close the figure to free up memory
    else:
        st.write("No data found for the selected cluster.")
