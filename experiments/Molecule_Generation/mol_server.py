################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    raise ImportError("Please install the rdkit package to use this module.")
import json
import os
from loguru import logger
from charge.Experiment import Experiment
from charge.servers.server_utils import args
from mcp.server.fastmcp import FastMCP
from charge.clients.autogen import AutoGenClient
from charge.clients.Client import Client
import asyncio
from charge.servers import SMILES_utils as hf
import argparse

mcp = FastMCP(
    "SMILES Diagnosis and retrieval MCP Server",
    port=args.port,
    website_url=f"{args.host}",
)

MODEL = "gpt-oss:latest"
BACKEND = "ollama"
API_KEY = None
KWARGS = {}
JSON_FILE_PATH = f"{os.getcwd()}/known_molecules.json"


class DiagnoseSMILESTask(Experiment):
    def __init__(self):
        system_prompt = (
            "You are a world-class chemist. Your task is to diagnose and evaluate "
            "the quality of the provided SMILES strings. You will be given invalid"
            " SMILES strings, and your task is to identify the issues with them, "
            "correct them if possible. Return concise explanations for each SMILE string."
        )

        user_prompt = (
            "Diagnose the followig SMILES string {0}. Give it a short and concise "
            "explanation of what is wrong with it, and if possible, provide a corrected "
            "version of the SMILES string. If the SMILES string is valid, simply state "
            "'The SMILES string is valid.'"
        )
        super().__init__(system_prompt=system_prompt, user_prompt=user_prompt)

    def update_user_prompt(self, smiles: str) -> str:
        self.user_prompt.format(smiles)


@mcp.tool()
def diagnose_smiles(smiles: str) -> str:
    """
    Diagnose a SMILES string. Returns a diagnosis of the SMILES string.

    Args:
        smiles (str): The input SMILES string.
    Returns:
        str: The diagnosis of the SMILES string.
    """
    experiment = DiagnoseSMILESTask()
    experiment.update_user_prompt(smiles)
    diagnose_agent = AutoGenClient(
        experiment_type=experiment,
        model=MODEL,
        backend=BACKEND,
        api_key=API_KEY,
        model_kwargs=KWARGS,
    )

    try:
        response = asyncio.run(diagnose_agent.run())
        assert response is not None
        assert len(response.messages) > 0
        assert response.messages[-1] is not None
        return f"SMILES diagnoses: {response.messages[-1].content}"

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return "Error: Unable to process the SMILES string at this time."


@mcp.tool()
def is_already_known(smiles: str) -> bool:
    """
    Check if a SMILES string provided is already known. Only provide
    valid SMILES strings. Returns True if the SMILES string is valid, and
    already in the database, False otherwise.
    Args:
        smiles (str): The input SMILES string.
    Returns:
        bool: True if the SMILES string is valid and known, False otherwise.

    Raises:
        ValueError: If the SMILES string is invalid.
    """
    if not Chem.MolFromSmiles(smiles):
        raise ValueError("Invalid SMILES string.")

    try:
        canonical_smiles = hf.canonicalize_smiles(smiles)

        try:
            with open(JSON_FILE_PATH) as f:
                known_mols = json.load(f)
                known_smiles = [mol["smiles"] for mol in known_mols]

        except FileNotFoundError:
            logger.warning(f"{JSON_FILE_PATH} not found. Creating a new one.")
            known_mols = []

    except Exception as e:
        raise ValueError("Error in canonicalizing SMILES string.") from e

    # Check if the SMILES string is already known (in the database)
    # This is a placeholder for the actual database check
    return canonical_smiles in known_smiles


# Add the SMILES utility functions as MCP tools
mcp.tool()(hf.canonicalize_smiles)
mcp.tool()(hf.verify_smiles)
mcp.tool()(hf.get_synthesizability)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Molecule Tools Server")
    Client.add_std_parser_arguments(parser)

    args = parser.parse_args()
    # global MODEL, BACKEND, API_KEY, KWARGS, JSON_FILE_PATH
    MODEL = args.model if args.model else MODEL
    BACKEND = args.backend if args.backend else BACKEND
    MODEL, BACKEND, API_KEY, KWARGS = AutoGenClient.configure(
        model=MODEL, backend=BACKEND
    )
    logger.info(f"Using model: {MODEL} on backend: {BACKEND}")
    JSON_FILE_PATH = args.json_file if args.json_file else JSON_FILE_PATH
    logger.info(f"Using known molecules database at: {JSON_FILE_PATH}")

    mcp.run(transport="sse")
