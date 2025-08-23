---
layout: post
title: "Messages"
categories: [RAG]
---
# Messages

## Overview 

- Messages are the unit of communication in chat models.
- used to represent the input and output of a chat model, as well as any additional context or metadata.

- each message consists of:
1. Role: The role of the message(eg., 'use', "assistant")
2. Context: The content of the message
3. Additional metadata: id, name,token usage etc

## Role
* Roles are used to distinguish between different types of messages in a conversation and help the chat model understand how to respond to a given sequence of messages.

 * System: used to tell the chat model how to behave and provide additional context. 
 * user : Represents input from a user interacting with the model, usually in the form of text or other interactive input.
 * assistant: represents a response from the model, which can include text or a request to invoke tools.
 * tool: a message used to pass the results of a tool invocation back to the model after external data or processing has been retrieved.
 * function(legacy): This is a legacy role corresponding to OPenAI's legacy function-calling API.

## Content
- content of a message text or a list of dictionaries representing multimodal data.


    * SystemMessage -- for content which should be passed to direct the conversation
    * HumanMessage -- for content in the input from the user.
    * AIMessage -- for content in the response from the model.
    * Multimodality -- for more information on multimodal content.

* Other Message Data

    * Depending on the chat model provider, messages can include other data such as:

        * ID: An optional unique identifier for the message.
        * Name: An optional name property which allows differentiate between different entities/speakers with the same role. Not all models support this!
        * Metadata: Additional information about the message, such as timestamps, token usage, etc.
        * Tool Calls: A request made by the model to call one or more tools> See tool calling for more information.

## Conversation structure
- sequence of messages into a chat model should follow a specific structure to ensure that the chat model can generate a valid response.

## Langchain Messages
- LangChain provides a unified message format that can be used across all chat models, allowing users to work with different chat models without worrying about the specific details of the message format used by each model provider.

- LangChain messages are Python objects that subclass from a BaseMessage.

- The five main message types are:

    - SystemMessage: corresponds to system role
    - HumanMessage: corresponds to user role
    - AIMessage: corresponds to assistant role
    - AIMessageChunk: corresponds to assistant role, used for streaming responses
    - ToolMessage: corresponds to tool role
    - RemoveMessage -- does not correspond to any role. This is an abstraction, mostly used in LangGraph to manage chat history.
    - Legacy FunctionMessage: corresponds to the function role in OpenAI's legacy function-calling API.
## System Message

- A system message is used to prime the behaviour of the AI model and provide additional context, such as instructing the model to adopt a specific persona or setting the tone of the conversation.
- different chat providers support system message in one of the following ways:
    - Through a system message role: the message is included as part of the message sequence with the role explicitly set as "system"
    - Through a separate API parameter for system instructions: Instead of being included as a message, system instructions are passes via a dedicated API parameter.
    - No support for system messages

## HumanMessage
- corresponds to the "user" role. A human message represents input from a user interacting with the model.
## AIMessage
- It is used to represent a message with the role "assistant". This is the response from the model, which can include text or a request to invoke tools.

- Attributes:
    - content
    - tool_calls
    - invalid_tools_calls
    - usage_metadata
    - id
    - response_metadata
## AIMessageChunk
- It is common to stream responses for the chat model as they are being genearted, so the user can see the response in real-time instead of waiting for the entire response to be generated before displaying it.

- AIMessageChunks support the + operator to message them into a single AIMessage.

## ToolMessage
- This represents a message with role "tool", which contains the result of calling a tool.In addition to role and content, this message has:
    - tool_call_id: conveys the id of the call to the tool that was called to produce this result.
    - artifact: can be used to pass along arbitrary artifacts of the tool execution which are useful to track but which should not be sent to the model.

## RemoveMessage
- This is a special message type that does not correspond to any roles. It is used for managing chat history in LangGraph.

## Reference

[messages](https://python.langchain.com/docs/concepts/messages/#toolmessage)