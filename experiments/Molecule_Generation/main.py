import argparse
import asyncio

from LMOExperiment import LMOExperiment as LeadMoleculeOptimization
import os
from charge.clients.Client import Client
import logging
import helper_funcs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--lead-molecule", type=str, default="CC(=O)O[C@H](C)CCN")
parser.add_argument(
    "--client", type=str, default="autogen", choices=["autogen", "gemini"]
)

parser.add_argument(
    "--server-path",
    type=str,
    default="mol_server.py",
    help="Path to an existing MCP server script",
)

# Add standard CLI arguments
Client.add_std_parser_arguments(parser)

args = parser.parse_args()


if __name__ == "__main__":

    myexperiment = LeadMoleculeOptimization(lead_molecule=args.lead_molecule)
    server_path = args.server_path
    assert server_path is not None, "Server path must be provided"

    if args.client == "gemini":
        from charge.clients.gemini import GeminiClient

        client_key = os.getenv("GOOGLE_API_KEY")
        assert client_key is not None, "GOOGLE_API_KEY must be set in environment"
        runner = GeminiClient(experiment_type=myexperiment, api_key=client_key)
    elif args.client == "autogen":
        from charge.clients.autogen import AutoGenClient

        (model, backend, API_KEY, kwargs) = AutoGenClient.configure(
            args.model, args.backend
        )

        runner = AutoGenClient(
            experiment_type=myexperiment,
            model=model,
            backend=backend,
            api_key=API_KEY,
            model_kwargs=kwargs,
            server_path=server_path,
        )

    lead_molecule_smiles = args.lead_molecule
    logger.info(f"Starting experiment with lead molecule: {lead_molecule_smiles}")
    parent_id = 0
    node_id = 0
    lead_molecule_data = helper_funcs.post_process_smiles(
        smiles=lead_molecule_smiles, parent_id=parent_id - 1, node_id=node_id
    )

    # Start the db with the lead molecule
    helper_funcs.save_list_to_json_file(
        data=[lead_molecule_data], file_path="known_molecules.json"
    )
    logger.info(f"Storing found molecules in known_molecules.json")

    # Run the experiment in a loop
    new_molecules = helper_funcs.get_list_from_json_file(
        file_path="known_molecules.json"
    )  # Start with known molecules

    mol_data = [lead_molecule_data]

    while True:
        try:
            results = asyncio.run(runner.run())
            results = results.as_list()  # Convert to list of strings
            logger.info(f"New molecules generated: {results}")
            processed_mol = helper_funcs.post_process_smiles(
                smiles=results[0], parent_id=parent_id, node_id=node_id
            )
            canonical_smiles = processed_mol["smiles"]
            if canonical_smiles not in new_molecules:
                new_molecules.append(processed_mol["smiles"])
                mol_data.append(processed_mol)
                helper_funcs.save_list_to_json_file(
                    data=mol_data, file_path="known_molecules.json"
                )
                logger.info(f"New molecule added: {canonical_smiles}")
                node_id += 1
            else:
                logger.info(f"Duplicate molecule found: {canonical_smiles}")

            logger.info(f"Total unique molecules so far: {new_molecules}")

            # TODO: Update the prompt with the new lead molecule or maybe the
            # the best performing molecule.

            # reset the runner for the next iteration
            runner.reset()

            if len(new_molecules) >= 5:  # Collect 5 new molecules then stop
                break
        except KeyboardInterrupt:
            logger.info("Experiment interrupted by user.")
            break
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            logger.info("Restarting the experiment...")
            runner.reset()
            continue

    logger.info(f"Experiment completed. Results: {new_molecules}")
