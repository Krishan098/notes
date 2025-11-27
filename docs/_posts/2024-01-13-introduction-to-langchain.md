---
layout: post
title: "Introduction to LangChain"
date: 2024-01-13
description: Notes on Introduction to LangChain
tags: [langchain]
categories: [LangChain]
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

# INTRO
- Framework for developing applications powered by LLMs.

- Development: LangChain

- Productionization: LangSmith

- Deployment: LangGraph

- LangChain implements a standard interface for large language models and related technologies, such as embedding models and vector stores, and integrates with hundreds of providers.

# Architecture:

- langchain-core: Base abstractions for chat models and other components.

- Integration packages (e.g. langchain-openai, langchain-anthropic, etc.): Important integrations have been split into lightweight packages that are co-maintained by the LangChain team and the integration developers.

- langchain: Chains, agents, and retrieval strategies that make up an application's cognitive architecture.

- langchain-community: Third-party integrations that are community maintained.

- langgraph: Orchestration framework for combining LangChain components into production-ready applications with persistence, streaming, and other key features. 

## langchain-core:

- The interfaces for core components like chat models, vector stores, tools and more are defined here.

## langchain

- It contains chains and retrieval strategies that make up an application;s cognitive architecture. All agents, chains and retrieval strategies are not specific to any one integration, but rather generic across all integrations.

## Integration Packages

- Integration packages such as langchain-anthropic, langchain-openai etc.

- API Reference : reference for all langchain-x packages.

## langchain-community

- this contains third-party integrations. This contains integrations for various components (chat models, vector stores, tools etc.)

## langgraph

- extension  of langchain aimed at building robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes in a graph.

## langserve

- package to deploy Langchain chains as REST APIs.

## LangSmith

- A developer platform that let's you drbug, test, evaluate and monitor LLM applications.