# Subgraphs
- A subgraph is a graph that is used as a node in another graph. 
- encapsulation applied to LangGraph.
- allows to build complex systems with multiple components that are themselves graphs.
![alt text](images/image-3.png)

- Some reasons for using subgraphs are:

1. building multi-agent systems
2. when you want to reuse a set of nodes in multiple graphs
3. when you want different teams to work on different parts of the graph independently, you can define each part as a subgraph, and as long as the subgraph interface (the input and output schemas) is respected, the parent graph can be built without knowing any details of the subgraph

## ways that the parent and subgraph communicate :

1. parent and subgraph have shared state keys in their state schemas.
- we can include the subgraph as a node in the parent graph.

2. parent and subgraph have different schema. 
- call the subgraph from inside a node in the parent graph.

For reference:
https://langchain-ai.github.io/langgraph/concepts/subgraphs/