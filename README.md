# Inventory-Genie

An AI-powered basic RAG application for managing orders and inventory. Provides real-time stock updates, checks availability, suggests similar products, and generates intelligent responses to both customer and manager queries using Gemini LLM.

## Features

- **Dynamic Inventory Management**: Easily monitor stock levels, track inventory changes, and manage warehouse data in real-time.
- **Real-Time Customer Support**: Utilize Gemini LLM to provide instant, accurate responses to customer queries, including order status and product availability.
- **Advanced Similarity Search**: Quickly find similar products using advanced embedding techniques and the FAISS library, enhancing product recommendations.
- **Comprehensive Data Integration**: Seamlessly integrate and process data from CSV files for orders, warehouse stock, and shipments to ensure smooth operations.

## Technologies Used

- **Streamlit**: Framework for creating interactive web applications.
- **Pandas**: Data manipulation and analysis.
- **LangChain**: For integrating and managing language model interactions.
- **Google Gemini LLM**: For generating responses based on customer and manager queries.
- **Sentence Transformers**: For generating embeddings and similarity search.
- **FAISS**: Facebook AI Similarity Search library for efficient nearest neighbor search.
- **NumPy**: For numerical operations and handling arrays.

## Results

### Customer Query

![image](https://github.com/user-attachments/assets/47ea2d4b-4ad9-434c-b991-ff177dab82ad)


### Manager Query

![image](https://github.com/user-attachments/assets/9647c100-8b23-4b36-919b-fd1996b8f2de)


## Future Improvements

1. **Predictive Analytics**: Implement machine learning models to forecast inventory needs and optimize stock levels.
2. **User Authentication**: Add user authentication and roles for better security and personalized access.
3. **Automated Notifications**: Integrate automated alerts and notifications for inventory levels, order statuses, and delivery updates.
   
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
