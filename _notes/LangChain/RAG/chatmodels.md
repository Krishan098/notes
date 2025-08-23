---
layout: default
title: "Chat Models"
categories: [RAG]
---
# Chat Models

- Chat models are language models that use a sequence of messages as inputs and return messages as outputs. 

## Overview 

- LLMs are advanced machine learning models that excel in a wide range of language-related tasks such as text generation, translation, summarization, question answering and more without needing task specific fine tuning for every scenario.

- Modern LLMs are typically accessed through a chat model interface that takes a list of messages as input and returns a message as output.


1. Tool calling: Many popular chat models offer a native tool calling API. This API allows developers to build rich applications that enable LLMs to interact with external services, APIs, and databases. Tool calling can also be used to extract structured information from unstructured data and perform various other tasks.
2. Structured output: A technique to make a chat model respond in a structured format, such as JSON that matches a given schema.
3. Multimodality: The ability to work with data other than text; for example, images, audio, and video.

## Features

- LangChain provides a consistent interface for working with chat models from different providers while offering additional features for monitoring, debugging, and optimizing the performance of applications that use LLMs.

1. Integrations with many chat model providers (e.g., Anthropic, OpenAI, Ollama, Microsoft Azure, Google Vertex, Amazon Bedrock, Hugging Face, Cohere, Groq). Please see chat model integrations for an up-to-date list of supported models.
2. Use either LangChain's messages format or OpenAI format.
3. Standard tool calling API: standard interface for binding tools to models, accessing tool call requests made by models, and sending tool results back to the model.
4. Standard API for structuring outputs via the with_structured_output method.
5. Provides support for async programming, efficient batching, a rich streaming API.
6. Integration with LangSmith for monitoring and debugging production-grade applications based on LLMs.
7. Additional features like standardized token usage, rate limiting, caching and more.

## Interface

- LangChain chat models implement the BaseChatModel interface. Because BaseChatModel also implements the Runnable Interface, chat models support a standard streaming interface, async programming, optimized batching, and more. Please see the Runnable Interface for more details.

- Many of the key methods of chat models operate on messages as input and return messages as output.

- Chat models offer a standard set of parameters that can be used to configure the model. These parameters are typically used to control the behavior of the model, such as the temperature of the output, the maximum number of tokens in the response, and the maximum time to wait for a response. 

## key methods:

1. invoke: the primary method for interacting with a chat model. takes a list of messages and then returns a list of messages

2. stream: allows you to stream the output of a chat model as it is generated.

3. bind_tools: A method that allows you to bind a tool to a chat model for usr in the model's execution context. 

4. batch: allows you to batch multiple requests to a chat model together for more efficient processing

5. with_structured_output: A wrapper around invoke method for models that natively support structured output.

## Context window

- A chat model's context window refers to the maximum size of the input sequence the model can process at one time. 

- the size of the input is measured in tokens which are the unit of processing th e model uses.

## Rate-limiting

- many chat model providers impose a limit on the number of requests that can be made in a given time period/

avoid it by:

1. avoid hitting rate limits by spacing out requests. rate_limiter parameter

2. try to recover from rate limit errors

3. fallback to another chat model

## Caching

- Chat model APIs can be slow, so we can cache the results of previous conversations. 

- An alternative approach is to use semantic caching, where you cache responses based on the meaning of the input rather than the exact input itself. This can be effective in some situations, but not in others.

- A semantic cache introduces a dependency on another model on the critical path of your application (e.g., the semantic cache may rely on an embedding model to convert text to a vector representation), and it's not guaranteed to capture the meaning of the input accurately.

- However, there might be situations where caching chat model responses is beneficial. For example, if you have a chat model that is used to answer frequently asked questions, caching responses can help reduce the load on the model provider, costs, and improve response times.

## Reference

https://python.langchain.com/docs/concepts/chat_models/