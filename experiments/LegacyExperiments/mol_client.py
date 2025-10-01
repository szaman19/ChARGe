################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import os
from contextlib import AsyncExitStack

client_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=client_key)

server_params = StdioServerParameters(
    command="python3", args=["mol_server.py"], env=None
)


async def main():

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            system_prompt = (
                "You are a helpful SMILES generator "
                + "that can generate new molecules in a SMILES format and optimize"
                + " for density and synthetic accessibility."
            )
            print("System prompt:", system_prompt)

            print("Prompt: \n")
            chat_prompt = (
                "Generate 5 new SMILES strings"
                + " for molecules similar to CC(=O)O[C@H](C)CCN. For each molecules you suggest "
                + " verify the SMILES,"
                + " check if it already known, and calculate its density and"
                + " synthetic accessibility. Only return molecules with higher density and"
                + " the same or lower synthetic accessibility."
                + " If a molecule is known or doesn't fit the criteria, move on and"
                + " generate a different one and try again."
                + " Output a list of the unique molecules \n\n"
            )

            print(chat_prompt)
            prompt = chat_prompt

            await session.initialize()

            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash-latest",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],
                    system_instruction=system_prompt,
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True, thinking_budget=8192
                    ),
                ),
            )
            assert response is not None
            assert response.usage_metadata is not None
            print("Thoughts tokens:", response.usage_metadata.thoughts_token_count)
            print("Output tokens:", response.usage_metadata.candidates_token_count)
            assert response.candidates is not None
            assert len(response.candidates) > 0
            assert response.candidates[0] is not None
            assert response.candidates[0].content is not None
            assert response.candidates[0].content.parts is not None

            print(response.candidates[0].content.parts)
            for part in response.candidates[0].content.parts:
                if not part.text:

                    continue
                if part.thought:
                    print("Thought summary:")
                    print(part.text)
                    print()
                else:
                    print("Answer:")
                    print(part.text)
                    print()
            print("Response text:", response.text)
            print("Response usage metadata:", response.usage_metadata)


if __name__ == "__main__":
    asyncio.run(main())
