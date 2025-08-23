# Memory
- Memory is a cognitive function that allows people to store, retrieve, and use information to understand their present and future. Consider the frustration of working with a colleague who forgets everything you tell them, requiring constant repetition! As AI agents undertake more complex tasks involving numerous user interactions, equipping them with memory becomes equally crucial for efficiency and user satisfaction. With memory, agents can learn from feedback and adapt to users' preferences. This guide covers two types of memory based on recall scope:

1. Short-term memory, or thread-scoped memory, can be recalled at any time from within a single conversational thread with a user. LangGraph manages short-term memory as a part of your agent's state. State is persisted to a database using a checkpointer so the thread can be resumed at any time. Short-term memory updates when the graph is invoked or a step is completed, and the State is read at the start of each step.

2. Long-term memory is shared across conversational threads. It can be recalled at any time and in any thread. Memories are scoped to any custom namespace, not just within a single thread ID. LangGraph provides stores (reference doc) to let you save and recall long-term memories.

![alt text](images/image-7.png)

refernce
https://langchain-ai.github.io/langgraph/concepts/memory/#what-is-memory