################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
try:
    from autogen_core.models import (
        AssistantMessage,
    )
except ImportError:
    raise ImportError(
        "Please install the autogen-agentchat package to use this module."
    )

def thoughts_callback(assistant_message):
    # print("In callback:", assistant_message)
    if assistant_message.type == "UserMessage":
        print(f"User: {assistant_message.content}")
    elif assistant_message.type == "AssistantMessage":

        if assistant_message.thought is not None:
            print(f"Model thought: {assistant_message.thought}")
        if isinstance(assistant_message.content, list):
            for item in assistant_message.content:
                if hasattr(item, "name") and hasattr(item, "arguments"):
                    print(
                        f"Function call: {item.name} with args {item.arguments}"
                    )
                else:
                    print(f"Model: {item}")
    elif assistant_message.type == "FunctionExecutionResultMessage":

        for result in assistant_message.content:
            if result.is_error:
                print(
                    f"Function {result.name} errored with output: {result.content}"
                )
            else:
                print(f"Function {result.name} returned: {result.content}")
    else:
        print("Model: ", assistant_message.message.content)
