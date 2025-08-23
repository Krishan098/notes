---
layout: default
title: "Tools"
categories: [RAG]
---
# TOOLS
- The tool abstraction in LangChain associates a Python function with a schema that defines the function's name, description and expected arguments.

## Key Concepts:
- Tools are a way to encapsulate a function and its schema in a way that can be passed to a chat model.
- Create tools using the @tool decorator, which simplifies the process of tool creation, supporting the following: 
    - Automatically infer the tool's name, description and expected arguments, while also supporting customization.
    - Defining tools that return artifacts.
    - Hiding input arguments from the schema using injected tool arguments.
## Tool Interface:
- key attributes:
    - name: name of the tool
    - description: a description of what the tool does.
    - args: Property that returns the JSON schema for the tool's arguments.
- key methods to execute function associated with the tool:
    - invoke: invoke the tool with given arguments.
    - ainvoke: invokes the tool with the given arguments, asynchronously.
## Tool Artifacts:
- Tools are utilities that can be called by a model, and whose outputs are designed to be fed back to a model. Sometimes, however, there are artifacts of a tool's execution that we want to make accessible to downstream components in our chain or agent, but that we don't want to expose to the model itself. For example if a tool returns a custom object, a dataframe or an image, we may want to pass some metadata about this output to the model without passing the actual output to the model. At the same time, we may want to be able to access this full output elsewhere, for example in downstream tools.

### Reference
[tools](https://python.langchain.com/docs/concepts/tools/)