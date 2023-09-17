This is some playground code to learn how langchain interacts with LLMs.
The basic goal was to build a conversational document interrogator
The idea was drop a doc, or set of docs, into a directory and just ask questons.

Things I've learned. 
Various models had different endpoints.  Each of these endpoints require a different langchain
llm/chat model end point.  e.g.:
    v1/completions -> OpenAI()
    /v1/chat/completion -> ChatOpenAI()

Each end point only supports certain models.  e.g:
    /v1/chat/completion supports GPT-4, GPT-3.5
    /v1/completion supports GPT-3, GPT base

NOTE: To get pytorch to work with CUDA (12.2) had to use the following
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

