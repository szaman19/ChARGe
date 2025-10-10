################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP
from charge.servers.SMARTS_reactions import SMARTS_mcp
from charge.servers.SMILES_utils import verify_smiles, canonicalize_smiles
import click

template_free_mcp = FastMCP("template_free_reaction_server")

template_free_mcp.tool()(verify_smiles)
template_free_mcp.tool()(canonicalize_smiles)


if __name__ == "__main__":
    click.echo("Starting SMARTS reaction server...")

    @click.command()
    @click.argument(
        "--exp_type",
        default="template",
        type=click.Choice(["template", "template-free"]),
    )
    def run_reaction_server(exp_type):
        if exp_type == "template":

            SMARTS_mcp.run(
                transport="sse",
            )
        elif exp_type == "template-free":
            template_free_mcp.run(
                transport="sse",
            )
        else:
            raise ValueError(f"Unknown experiment type: {exp_type}")
