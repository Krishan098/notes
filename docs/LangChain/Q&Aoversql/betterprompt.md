---
layout: default
title: "Better prompts"
parent: Q&A Over SQL
grand_parent: LangChain
nav_order: 2
---
# How to better prompt when doing SQL question-answering

## Dialect specific prompting

- One of the simplest things we can do is make our prompt specific to the SQL dialect we're using.

## Few-shot examples

- Including examples of natural language questions being converted to valid SQL queries against our database in the prompt will often improve model performance, especially for complex queries.

## Dynamic few-shot examples 

- If we have enough examples, we may want to only include the most relevant ones in the prompt, either because they don't fit in the model's context window or because the long tail of examples distracts the model. And specifically, given any input we want to include the examples most relevant to that input.

- We can do just this using an ExampleSelector. In this case we'll use a SemanticSimilarityExampleSelector, which will store the examples in the vector database of our choosing. At runtime it will perform a similarity search between the input and our examples, and return the most semantically similar ones.