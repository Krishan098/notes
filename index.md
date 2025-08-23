---
layout: home
title: "Neural Notes"
---

# Neural Notes

This is a collection of few of the things I have studied so far. Will try to add other things as well and there would be regular updates here.

---
- [All Notes](/notes/)
<ul>
  {% for note in site.notes %}
    <li>
      <a href="{{ note.url }}">{{ note.title }}</a>
    </li>
  {% endfor %}
</ul>