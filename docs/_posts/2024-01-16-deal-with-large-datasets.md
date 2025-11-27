---
layout: post
title: "Deal with large datasets"
date: 2024-01-16
description: Notes on Deal with large datasets
tags: [langchain, q&a-over-sql]
categories: [LangChain, Q&A Over SQL]
---

# How to deal with large databases when doing SQL question answering

- When there are many tables, columns, and/or high-cardinality columns, it becomes impossible for us to dump the full information about our database in every prompt. Instead, we must find ways to dynamically insert into the prompt only the most relevant information.

- In this guide we demonstrate methods for identifying such relevant information, and feeding this into a query-generation step. We will cover:

  1. Identifying a relevant subset of tables;

    2.  Identifying a relevant subset of column values.

## Many tables

- One of the main peices of info we need to include in our prompt is the schemas of the relevant tables. 

- When we have very many tables, we can't fit all of the schemas in a single prompt.

- so we extract the names of the tables related to the user input and then include only their schemas. 

- Easy way to do it is using tool-calling.

## High cardinality columns

- In order to filter columns that contain proper nouns such as addresses etc , we first need to double-check the spelling in order to filter the data correctly.

- One strategy is to create a vector store with all the distinct proper nowns that exist in the database. We can then query the vector store each user input and inject the most relevant proper nouns into the prompt.