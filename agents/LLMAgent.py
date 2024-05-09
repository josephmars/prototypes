from langchain.schema import HumanMessage, SystemMessage


base_prompt = """You are a helpful assistant what anwers the questions provided to you. If you don't know the answer, just say that you don't know, don't try to make up an answer."""
example_prompt = """Here are some examples of questions and answers that you can use to help you answer the questions: \n{examples}"""

class LLMAgent:
    def __init__(self, llm, prompt = base_prompt, examples = ""):
        self.prompt = prompt
        self.examples = examples
        self.llm = llm
        
        if self.examples != "":
            self.prompt = prompt + "\n" + example_prompt.format(examples = self.examples)
    
    def chat(self, query):
        messages = [SystemMessage(content= self.prompt), HumanMessage(content=query)]
        return(self.llm.invoke(messages).content)
    
    def print_prompt(self):
        return self.prompt
    