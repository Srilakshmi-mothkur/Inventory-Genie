import streamlit as st
import pandas as pd
from datetime import timedelta
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

def load_data():
    uploaded_files = st.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)
    dfs = {}
    for file in uploaded_files:
        df_name = file.name.split('.')[0].lower()
        dfs[df_name] = pd.read_csv(file)
    return dfs.get('orders'), dfs.get('warehouse'), dfs.get('shipments')

def check_availability(order_data, df_warehouse, df_shipments):

    product_id = order_data["product_id"]
    quantity_ordered = order_data["quantity_ordered"]

    warehouse_stock = df_warehouse[df_warehouse["product_id"] == product_id]["quantity_in_stock"].values[0]

    if quantity_ordered <= warehouse_stock:
        available_quantity = quantity_ordered
        delayed_quantity = 0
        expected_delivery_date = df_shipments[df_shipments["order_id"] == order_data["order_id"]][
            "expected_delivery_date"
        ].values[0]
        df_warehouse.loc[df_warehouse["product_id"] == product_id, "quantity_in_stock"] -= quantity_ordered

    else:
        available_quantity = warehouse_stock
        delayed_quantity = quantity_ordered - warehouse_stock
        df_warehouse.loc[df_warehouse["product_id"] == product_id, "quantity_in_stock"] = 0
        creation_time_per_product = 2  
        additional_creation_time = int(delayed_quantity * creation_time_per_product)
        available_date = df_warehouse[df_warehouse["product_id"] == product_id]["available_date"].values[0]
        available_date = pd.to_datetime(df_warehouse[df_warehouse["product_id"] == product_id]["available_date"].values[0])
        additional_seconds = additional_creation_time * 3600 
        expected_delivery_date = available_date + timedelta(seconds=additional_seconds)

    return available_quantity, delayed_quantity, expected_delivery_date


def handle_order_availability(order_data, df_orders, df_warehouse, df_shipments):
    llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="models/gemini-pro", temperature=0.7)

    prompt_template = PromptTemplate(
        input_variables=["quantity_ordered", "product_id", "available_quantity", "delayed_quantity", "expected_delivery_date"],
        template="""
        You are a customer service representative for an e-commerce company. 
        The customer has placed an order for {quantity_ordered} units of product {product_id}. 
        Please check the availability of the product in the warehouse and inform the customer 
        about the expected delivery date. If there's insufficient stock, inform the customer 
        about the available quantity and the delayed delivery for the remaining units. 
        Available quantity: {available_quantity}, Delayed quantity: {delayed_quantity}, 
        Expected delivery date: {expected_delivery_date}.
        What is the appropriate response to the customer?
        """
    )
    
    available_quantity, delayed_quantity, expected_delivery_date = check_availability(order_data, df_warehouse, df_shipments)
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )

    response = chain.run({
        "quantity_ordered": order_data["quantity_ordered"],
        "product_id": order_data["product_id"],
        "available_quantity": available_quantity,
        "delayed_quantity": delayed_quantity,
        "expected_delivery_date": expected_delivery_date,
    })

    return response

def generate_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    product_descriptions = df.apply(lambda row: f"Product ID: {row['product_id']}, Name: {row['product_name']}, Stock: {row['quantity_in_stock']}, Machine No: {row['machine_no']}, Available Date: {row['available_date']}", axis=1).tolist()
    product_embeddings = model.encode(product_descriptions)
    np.save('product_embeddings.npy', product_embeddings)

def search_similar_products(query_embedding, product_embeddings, top_k=5):
    dimension = product_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(product_embeddings)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return indices[0]

def generate_llm_response(query, df_warehouse):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    product_embeddings = np.load('product_embeddings.npy')
    query_embedding = model.encode([query])[0]
    similar_product_indices = search_similar_products(query_embedding, product_embeddings)
    similar_products = df_warehouse.iloc[similar_product_indices]
    
    prompt = f"Customer asked: {query}\n\n"
    prompt += "Similar products found:\n"
    for index, row in similar_products.iterrows():
        prompt += f"Product ID: {row['product_id']}, Name: {row['product_name']}, Stock: {row['quantity_in_stock']}, Available Date: {row['available_date']}\n"

    llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="models/gemini-pro", temperature=0.7)
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="{prompt}"
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )
    response = chain.run({
        "prompt": prompt
    })

    return response

def format_response(response):
    return "\n".join(response.split(". "))

def main():
    st.title('Order and Inventory Management System')

    df_orders, df_warehouse, df_shipments = load_data()

    if df_orders is not None and df_warehouse is not None and df_shipments is not None:
        st.sidebar.success("Files successfully uploaded!")
    else:
        st.sidebar.warning("Upload CSV files to proceed.")

    st.header("Customer Segment")
    if df_orders is not None and df_warehouse is not None and df_shipments is not None:
        order_id = st.number_input("Enter Order ID:", min_value=int(df_orders['order_id'].min()), max_value=int(df_orders['order_id'].max()))
        if st.button("Check Stock Availability"):
            order_data = df_orders[df_orders['order_id'] == order_id].iloc[0].to_dict()
            response = handle_order_availability(order_data, df_orders, df_warehouse, df_shipments)
            formatted_response = format_response(response)
            st.text(f"Customer Response:\n{formatted_response}")

    st.header("Manager Segment")
    if df_orders is not None and df_warehouse is not None and df_shipments is not None:
        query = st.text_input("Enter Query:")
        if st.button("Generate Response"):
            response = generate_llm_response(query, df_warehouse)
            formatted_response = format_response(response)
            st.text(f"Manager Response:\n{formatted_response}")

if __name__ == "__main__":
    main()
