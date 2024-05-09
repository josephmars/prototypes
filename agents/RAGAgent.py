
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma



base_prompt = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. 
    Context:
    {context}
    
    Question: {question}
    
    Helpful Answer:"""

example_prompt = """Here are some examples of questions and answers that you can use to help you answer the questions: {examples}"""


class RAGAgent:
    def __init__(self, llm, splitted_document, embeddings, prompt = base_prompt, examples = "", additional_args = {}):
        self.prompt = prompt
        self.examples = examples
        self.llm = llm
        self.splitted_document = splitted_document
        self.embeddings = embeddings
        self.output = None
        if self.examples != "":
            self.prompt = prompt + "\n" + example_prompt.format(examples = self.examples)
            
        if additional_args is not None:
            for key in additional_args:
                self.prompt = self.prompt.replace("{" + key + "}", additional_args[key])
    
    def chat(self, query, return_source_documents = True):
        
        db = Chroma.from_documents(self.splitted_document, self.embeddings)


        # Build prompt
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(self.prompt)

        # Run chain
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=db.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        result = qa_chain({"query": query})
        answer = result["result"]
        
        output = {"answer": answer}
        

        output[ "source_documents"] = result["source_documents"]
        output["prompt"] = QA_CHAIN_PROMPT
        
        self.output = output
        
        if return_source_documents:
            return output
        else:
            return output["answer"]

    
    def get_prompt(self):
        return self.prompt
    
    def get_output(self):
        return self.output
    
    
