---
layout: post
title: "Deep Research"
date: 2024-01-04
description: Notes on Deep Research
tags: [machine-learning]
categories: [Machine Learning]
---

- It's a new agentic capability that conducts multi-step research on the internet for complex tasks.

- It can do work for us independently- when given a prompt, it will find, analyze and synthesize hundreds of online sources to create a comprehensive report at the level of a research analyst. 

- It has been trained using end-to-end reinforcement learning on hard browsing and reasoning tasks across a range of domains.

- Through that training, it learned to plan and execute a multi-step trajectory to find the data it needs, backtracking and reacting to real-time information where necessary. 

- The model is also able to browse over user uploaded files, plot and iterate on graphs using the python tool, embed both generated graphs and images from websites in its responses, and cite specific sentences or passages from its sources.

## Humanity's Last Exam

- Consists of 2500 challenging questions across over a hundred subjects.

## Agent Frameworks

- code agent.

- letting the agent express its actions in code has several advantages, but most notably that **code is specifically designed to express complex sequences of actions.**

- Code actions are much more concise than JSON.

- Code enables to re-use tools from common libraries.

- Better performance in benchmarks, due to two reasons:

    - More intuitive way to express actions
    - Extensive exposure of LLMs to code in training

## Making the right tools

1. A web browser: A fully fledged web browser interction like Operator will be needed to reach full performance.

2. A simple text inspector, to be able to read a bunch of text file format.

- Operator is an agent that can go to the web to perform tasks for us. It uses its own browser, it can look at a webpage and interact with it by typin, clicking and scrolling.