---
layout: post
title: "Tool Calling"
date: 2024-02-03
description: Notes on Tool Calling
tags: [langchain, rag]
categories: [LangChain, RAG]
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

# Tool Calling/Function calling

- Many AI applications interact directly with humans. In these cases, it is appropriate for models to respond in natural language. But what about cases where we want a model to also interact directly with systems, such as databases or an API? These systems often have a particular input schema; for example, APIs frequently have a required payload structure. This need motivates the concept of tool calling. 

![](/notes/assets/img/posts/image-13.png)

## Key concepts:

1. Tool Creation: use the @tool decorator to create a tool. A tool is an association between a function and it's schema.

2. Tool Binding: The tool needs to be connected to a model that supports tool calling. This gives the model awareness of the tool and the associated input schema required by the tool.

3. Tool Calling: When appropriate, the model can decide to call a tool and ensure its response conforms the tool's input schema. 

4. Tool Execution: The tool can be executed using the arguments provided by the model.

![](/notes/assets/img/posts/image-14.png)

## Tool calling
![](/notes/assets/img/posts/image-15.png)

- A key principle of tool calling is that the model decides when to use a tool based on the input's relevance. The model doesn't always need to call a tool.

## Tool execution
- they can be directly invoked

## Reference
[tool-calling](https://python.langchain.com/docs/concepts/tool_calling/)
[function-calling](https://platform.openai.com/docs/guides/function-calling/example-use-cases?api-mode=responses)