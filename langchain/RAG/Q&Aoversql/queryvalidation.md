# How to do query validation as part of SQL question-answering

## Query checker

- Perhaps the simplest strategy is to ask the model itself to check the original query for common mistakes

- The obvious downside of this approach is that we need to make two model calls instead of one to generate our query.

## Human-in-the-loop
- in Q&Aoversql.md
## Error handling

- At some point, the model will make a mistake and craft an invalid SQL query. Or an issue will arise with our database. Or the model API will go down. We'll want to add some error handling behavior to our chains and agents so that we fail gracefully in these situations, and perhaps even automatically recover.

### try/except tool call

- ```sh
    from typing import Any

    from langchain_core.runnables import Runnable, RunnableConfig


    def try_except_tool(tool_args: dict, config: RunnableConfig) -> Runnable:
    try:
        complex_tool.invoke(tool_args, config=config)
    except Exception as e:
        return f"Calling tool with arguments:\n\n{tool_args}\n\nraised the following error:\n\n{type(e)}: {e}"


    chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | try_except_tool

    print(
    chain.invoke(
        "use complex tool. the args are 5, 2.1, empty dictionary. don't forget dict_arg"
    )
    )
    ```
### Fallbacks

- We can also try to fallback to a better model in the event of a tool invocation error

### Retry with exception

- To take things one step further, we can try to automatically re-run the chain with the exception passed in, so that the model may be able to correct its behavior