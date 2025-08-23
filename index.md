---
layout: default
title: "Neural Notes"
---

# Neural Notes

This is a collection of few of the things I have studied so far. Will try to add other things as well and there would be regular updates here.

## Categories

### Machine Learning & AI Research
{% assign ml_notes = site.notes | where_exp: "item", "item.path contains '/Machine Learning/'" %}
{% for note in ml_notes %}
- [{{ note.title | default: note.name | replace: ".md", "" | replace: "-", " " | capitalize }}]({{ note.url }})
{% endfor %}

### Code Compression to Reduce Hallucinations
{% assign compression_notes = site.notes | where_exp: "item", "item.path contains '/Code compression to reduce hallucinations/'" %}
{% for note in compression_notes %}
- [{{ note.title | default: note.name | replace: ".md", "" | replace: "-", " " | capitalize }}]({{ note.url }})
{% endfor %}

### LangChain & RAG
#### LangChain
{% assign langchain_notes = site.notes | where_exp: "item", "item.path contains '/LangChain/'" %}
{% for note in langchain_notes %}
- [{{note.title | default: note.name | replace: ".md", "" | replace: "-"," " | capitalize}}]({{note.url}})
{% endfor %}
#### RAG (Retrieval Augmented Generation)
{% assign rag_notes = site.notes | where_exp: "item", "item.path contains '/LangChain/RAG/'" %}
{% for note in rag_notes %}
- [{{ note.title | default: note.name | replace: ".md", "" | replace: "-", " " | capitalize }}]({{ note.url }})
{% endfor %}

#### Q&A over SQL
{% assign sql_notes = site.notes | where_exp: "item", "item.path contains '/LangChain/Q&Aoversql/'" %}
{% for note in sql_notes %}
- [{{ note.title | default: note.name | replace: ".md", "" | replace: "-", " " | capitalize }}]({{ note.url }})
{% endfor %}
---

