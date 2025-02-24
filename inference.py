import streamlit as st
from langchain.chains import LLMChain
from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
import numpy as np
import os
import torch
torch.set_num_threads(1)


def inference_faiss(chat_model, question, embedding_model_global, index, docstore, chat_history):
    st.write("Starting inference...")  # Debug point 11
    
    try:
        # Test retrieval
        query_embedding = embedding_model_global.embed_query(question)
        st.write("Created query embedding")  # Debug point 12
        
        k = 3
        D, I = index.search(np.array([query_embedding]), k=k)
        st.write(f"Retrieved {len(I[0])} documents")  # Debug point 13
        
        # Collect contexts
        contexts = []
        for i, idx in enumerate(I[0]):
            if idx != -1:
                doc = docstore.search(idx)
                if hasattr(doc, "page_content"):
                    contexts.append(doc.page_content)
                    st.write(f"Context {i+1}: {doc.page_content[:100]}...")  # Debug point 14
        
        if not contexts:
            return "No relevant context found in the documents."
            
        # Combine contexts
        context = "\n\n---\n\n".join(contexts)
        
        # Create chat completion
        chat_model = ChatTogether(
            together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
            model=chat_model,
        )
        
        prompt_template = PromptTemplate(
            input_variables=["history", "context", "question"],
            template="""You are a financial advisor specializing in Bajaj Finance Fixed Deposits. Use the following context to answer questions accurately:

            Context: {context}

            Chat History: {history}

            Question: {question}

            Answer to general conversation texts like hello,bye,etc
            **Strict Instructions to Avoid Hallucination:**
            1. Only answer using the provided context.
            2. Do not assume or generate information beyond what is explicitly mentioned in the context.
            3. Always quote numerical values, interest rates, and tenure periods exactly as found in the context.
            4. If multiple interest rates exist, specify whether they apply to general citizens or senior citizens.
            5. For yield-related questions, provide both the FD rate and yield percentage.
            6.If the question requires a numerical calculation (e.g., FD maturity, tax deduction), perform the necessary calculation.**
            7.Use the compound interest formula where required:**  
            [A = P * r * t
            ]
            where:  
            - P = Principal amount  
            - r = Interest rate (in decimal)  
            - n = Compounding frequency per year (1 for annual, 12 for monthly)  
            - t = Time in years  
            8. For tax-related queries**, apply TDS deduction rules:
            - If FD interest **exceeds ₹40,000 (₹50,000 for seniors)**, deduct **10% TDS**.
            - If PAN is missing, apply **20% TDS**.
            4. If multiple interest rates exist**, clearly specify whether they apply to **general citizens** or **senior citizens**.
            5. For yield-related questions, provide both FD rate and yield percentage.
            If the question asks for an interest rate for a **specific tenure (e.g., 37 months)**, but the provided information only contains **range-based tenures**, find the correct range and use the corresponding interest rate.
            2. Example: If the tenure is **37 months**, and the provided range is **36-60 months**, return the interest rate for **36-60 months**.
            3. If multiple ranges match, return the rate from the most relevant range.
            6. Maintain clarity and conciseness, avoiding unnecessary details.

            **Response:**"""
        )
        
        qa_chain = LLMChain(llm=chat_model, prompt=prompt_template)
        
        history_context = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
        )
        
        st.write("Generating response...")  # Debug point 15
        answer = qa_chain.run(
            history=history_context,
            context=context,
            question=question
        )
        
        return answer
        
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        return "An error occurred while processing your question."

def inference_pinecone(chat_model, question,embedding_model_global, pinecone_index,chat_history):
  import pinecone
  from pinecone import Pinecone
  from langchain_together import ChatTogether
  import numpy as np

  query_embedding = embedding_model_global.embed_query(question)
  query_embedding = np.array(query_embedding)

  search_results =  pinecone_index.query(
      vector=query_embedding.tolist(),
      top_k=4,
      include_metadata=True
  )
  contexts = [result['metadata']['text'] for result in search_results['matches']]

  context = "\n".join(contexts)

  history = "\n".join(
      [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
  )

  prompt = f"""You are a financial advisor specializing in Bajaj Finance Fixed Deposits. Use the following context to answer questions accurately:

            Context: {context}

            Chat History: {history}

            Question: {question}

            Answer to general conversation texts like hello,bye,etc
            **Strict Instructions to Avoid Hallucination:**
            1. Only answer using the provided context.
            2. Do not assume or generate information beyond what is explicitly mentioned in the context.
            3. Always quote numerical values, interest rates, and tenure periods exactly as found in the context.
            4. If multiple interest rates exist, specify whether they apply to general citizens or senior citizens.
            5. For yield-related questions, provide both the FD rate and yield percentage.
            6.If the question requires a numerical calculation (e.g., FD maturity, tax deduction), perform the necessary calculation.**
            7.Use the compound interest formula where required:**  
            [A = P * r * t
            ]
            where:  
            - P = Principal amount  
            - r = Interest rate (in decimal)  
            - n = Compounding frequency per year (1 for annual, 12 for monthly)  
            - t = Time in years  
            8. For tax-related queries**, apply TDS deduction rules:
            - If FD interest **exceeds ₹40,000 (₹50,000 for seniors)**, deduct **10% TDS**.
            - If PAN is missing, apply **20% TDS**.
            4. If multiple interest rates exist**, clearly specify whether they apply to **general citizens** or **senior citizens**.
            5. For yield-related questions, provide both FD rate and yield percentage.
            If the question asks for an interest rate for a **specific tenure (e.g., 37 months)**, but the provided information only contains **range-based tenures**, find the correct range and use the corresponding interest rate.
            2. Example: If the tenure is **37 months**, and the provided range is **36-60 months**, return the interest rate for **36-60 months**.
            3. If multiple ranges match, return the rate from the most relevant range.
            6. Maintain clarity and conciseness, avoiding unnecessary details.

            **Response:**"""

  llm = ChatTogether(api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
                  model=chat_model,  )

  response = llm.predict(prompt)
  print(response)
  return response


def inference(vectordb_name, chat_model, question,embedding_model_global,index,docstore,pinecone_index, chat_history):
    if vectordb_name == "FAISS":
        answer=inference_faiss(chat_model, question,embedding_model_global,index,docstore,chat_history)
        return answer
    elif vectordb_name == "Pinecone":
        answer=inference_pinecone(chat_model, question,embedding_model_global, pinecone_index,chat_history)
        return answer
    else:
        print("Invalid Choice")