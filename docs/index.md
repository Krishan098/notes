---
layout: home
title: "Neural Notes"
---

# Neural Notes

This is a collection of few of the things I have studied so far. Will try to add other things as well and there would be regular updates here.

<div class="stats">
    <div class="stat-item">
        <div class="stat-number">{{ site.notes.size }}</div>
        <div class="stat-label">Total Notes</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">4</div>
        <div class="stat-label">Categories</div>
    </div>
</div>

## Categories

<div class="category-card ml-section">

### Machine Learning & AI Research
{% assign ml_notes = site.notes | where_exp: "item", "item.path contains '/Machine Learning/'" %}
{% for note in ml_notes %}
- [{{ note.title | default: note.name | replace: ".md", "" | replace: "-", " " | capitalize }}]({{ note.url }})
{% endfor %}

</div>

<div class="category-card compression-section">

### Code Compression to Reduce Hallucinations
{% assign compression_notes = site.notes | where_exp: "item", "item.path contains '/Code compression to reduce hallucinations/'" %}
{% for note in compression_notes %}
- [{{ note.title | default: note.name | replace: ".md", "" | replace: "-", " " | capitalize }}]({{ note.url }})
{% endfor %}
</div>

### LangChain & RAG

<div class="langchain-section">

#### LangChain
{% assign langchain_notes = site.notes | where_exp: "item", "item.path contains '/LangChain/'" %}
{% for note in langchain_notes %}
- [{{note.title | default: note.name | replace: ".md", "" | replace: "-"," " | capitalize}}]({{note.url}})
{% endfor %}

</div>

<div class="rag-section">

#### RAG (Retrieval Augmented Generation)
{% assign rag_notes = site.notes | where_exp: "item", "item.path contains '/LangChain/RAG/'" %}
{% for note in rag_notes %}
- [{{ note.title | default: note.name | replace: ".md", "" | replace: "-", " " | capitalize }}]({{ note.url }})
{% endfor %}

</div>

<div class="sql-section">

#### Q&A over SQL
{% assign sql_notes = site.notes | where_exp: "item", "item.path contains '/LangChain/Q&Aoversql/'" %}
{% for note in sql_notes %}
- [{{ note.title | default: note.name | replace: ".md", "" | replace: "-", " " | capitalize }}]({{ note.url }})
{% endfor %}

</div>
---

