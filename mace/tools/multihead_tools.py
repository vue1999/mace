import argparse
import ast
import dataclasses
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from copy import deepcopy

import torch

from mace.cli.fine_tuning_select import (
    FilteringType,
    SelectionSettings,
    SubselectType,
    select_samples,
)
from mace.data import KeySpecification
from mace.tools.scripts_utils import SubsetCollection, get_dataset_from_xyz


@dataclasses.dataclass
class HeadConfig:
    head_name: str
    key_specification: KeySpecification
    train_file: Optional[Union[str, List[str]]] = None
    valid_file: Optional[Union[str, List[str]]] = None
    test_file: Optional[str] = None
    test_dir: Optional[str] = None
    E0s: Optional[Any] = None
    statistics_file: Optional[str] = None
    valid_fraction: Optional[float] = None
    config_type_weights: Optional[Dict[str, float]] = None
    keep_isolated_atoms: Optional[bool] = None
    atomic_numbers: Optional[Union[List[int], List[str]]] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    avg_num_neighbors: Optional[float] = None
    compute_avg_num_neighbors: Optional[bool] = None
    collections: Optional[SubsetCollection] = None
    train_loader: Optional[torch.utils.data.DataLoader] = None
    z_table: Optional[Any] = None
    atomic_energies_dict: Optional[Dict[str, float]] = None


def dict_head_to_dataclass(
    head: Dict[str, Any], head_name: str, args: argparse.Namespace
) -> HeadConfig:
    """Convert head dictionary to HeadConfig dataclass."""
    # parser+head args that have no defaults but are required
    if (args.train_file is None) and (head.get("train_file", None) is None):
        raise ValueError(
            "train file is not set in the head config yaml or via command line args"
        )

    return HeadConfig(
        head_name=head_name,
        train_file=head.get("train_file", args.train_file),
        valid_file=head.get("valid_file", args.valid_file),
        test_file=head.get("test_file", None),
        test_dir=head.get("test_dir", None),
        E0s=head.get("E0s", args.E0s),
        statistics_file=head.get("statistics_file", args.statistics_file),
        valid_fraction=head.get("valid_fraction", args.valid_fraction),
        config_type_weights=head.get("config_type_weights", args.config_type_weights),
        compute_avg_num_neighbors=head.get(
            "compute_avg_num_neighbors", args.compute_avg_num_neighbors
        ),
        atomic_numbers=head.get("atomic_numbers", args.atomic_numbers),
        mean=head.get("mean", args.mean),
        std=head.get("std", args.std),
        avg_num_neighbors=head.get("avg_num_neighbors", args.avg_num_neighbors),
        key_specification=head["key_specification"],
        keep_isolated_atoms=head.get("keep_isolated_atoms", args.keep_isolated_atoms),
    )


def prepare_default_head(args: argparse.Namespace) -> Dict[str, Any]:
    """Prepare a default head from args."""
    return {
        "Default": {
            "train_file": args.train_file,
            "valid_file": args.valid_file,
            "test_file": args.test_file,
            "test_dir": args.test_dir,
            "E0s": args.E0s,
            "statistics_file": args.statistics_file,
            "key_specification": args.key_specification,
            "valid_fraction": args.valid_fraction,
            "config_type_weights": args.config_type_weights,
            "keep_isolated_atoms": args.keep_isolated_atoms,
        }
    }


def prepare_pt_head(
    args: argparse.Namespace,
    pt_keyspec: KeySpecification,
    foundation_model_num_neighbours: float,
) -> Dict[str, Any]:
    """Prepare a pretraining head from args."""
    if (
        args.foundation_model in ["small", "medium", "large"]
        or args.pt_train_file == "mp"
    ):
        logging.info(
            "Using foundation model for multiheads finetuning with Materials Project data"
        )
        pt_keyspec.update(
            info_keys={"energy": "energy", "stress": "stress"},
            arrays_keys={"forces": "forces"},
        )
        pt_head = {
            "train_file": "mp",
            "E0s": "foundation",
            "statistics_file": None,
            "key_specification": pt_keyspec,
            "avg_num_neighbors": foundation_model_num_neighbours,
            "compute_avg_num_neighbors": False,
        }
    else:
        pt_head = {
            "train_file": args.pt_train_file,
            "valid_file": args.pt_valid_file,
            "E0s": "foundation",
            "statistics_file": args.statistics_file,
            "valid_fraction": args.valid_fraction,
            "key_specification": pt_keyspec,
            "avg_num_neighbors": foundation_model_num_neighbours,
            "keep_isolated_atoms": args.keep_isolated_atoms,
            "compute_avg_num_neighbors": False,
        }

    return pt_head


def assemble_mp_data(
    args: argparse.Namespace,
    head_config_pt: HeadConfig,
    tag: str,
) -> SubsetCollection:
    """Assemble Materials Project data for fine-tuning."""
    try:
        checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mp_traj_combined.xyz"
        cache_dir = (
            Path(os.environ.get("XDG_CACHE_HOME", "~/")).expanduser() / ".cache/mace"
        )
        checkpoint_url_name = "".join(
            c for c in os.path.basename(checkpoint_url) if c.isalnum() or c in "_"
        )
        cached_dataset_path = f"{cache_dir}/{checkpoint_url_name}"
        if not os.path.isfile(cached_dataset_path):
            os.makedirs(cache_dir, exist_ok=True)
            # download and save to disk
            logging.info("Downloading MP structures for finetuning")
            _, http_msg = urllib.request.urlretrieve(
                checkpoint_url, cached_dataset_path
            )
            if "Content-Type: text/html" in http_msg:
                raise RuntimeError(
                    f"Dataset download failed, please check the URL {checkpoint_url}"
                )
            logging.info(f"Materials Project dataset to {cached_dataset_path}")
        output = f"mp_finetuning-{tag}.xyz"
        atomic_numbers = (
            ast.literal_eval(args.atomic_numbers)
            if args.atomic_numbers is not None
            else None
        )
        settings = SelectionSettings(
            configs_pt=cached_dataset_path,
            output=f"mp_finetuning-{tag}.xyz",
            atomic_numbers=atomic_numbers,
            num_samples=args.num_samples_pt,
            seed=args.seed,
            head_pt="pbe_mp",
            weight_pt=args.weight_pt_head,
            filtering_type=FilteringType(args.filter_type_pt),
            subselect=SubselectType(args.subselect_pt),
            default_dtype=args.default_dtype,
        )
        select_samples(settings)
        head_config_pt.train_file = [output]
        collections_mp, _ = get_dataset_from_xyz(
            work_dir=args.work_dir,
            train_path=output,
            valid_path=None,
            valid_fraction=args.valid_fraction,
            config_type_weights=None,
            test_path=None,
            seed=args.seed,
            key_specification=head_config_pt.key_specification,
            head_name="pt_head",
            keep_isolated_atoms=args.keep_isolated_atoms,
        )
        return collections_mp
    except Exception as exc:
        raise RuntimeError(
            "Model or descriptors download failed and no local model found"
        ) from exc


def generate_pseudolabels_for_configs(model, configs, z_table, r_max, device, batch_size=32):
    """
    Generate pseudolabels for a list of Configuration objects.
    
    Args:
        model: The foundation model
        configs: List of Configuration objects
        z_table: Atomic number table
        r_max: Cutoff radius
        device: Device to run model on
        batch_size: Batch size for inference
        
    Returns:
        List of Configuration objects with updated properties
    """
    import torch
    import logging
    import copy
    import numpy as np
    from tqdm import tqdm
    import traceback
    
    if not configs:
        return []
    
    # Ensure model is in eval mode
    model.eval()
    
    # Create batches of configurations
    batches = [configs[i:i + batch_size] for i in range(0, len(configs), batch_size)]
    logging.info(f"Processing {len(configs)} configurations in {len(batches)} batches")
    
    updated_configs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc="Generating pseudolabels")):
            try:
                batch_inputs = []
                valid_configs = []  # Track which configs were successfully converted to AtomicData
                
                # Create AtomicData objects for each configuration
                for config in batch:
                    try:
                        from mace.data import AtomicData
                        data = AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
                        batch_inputs.append(data)
                        valid_configs.append(config)
                    except Exception as e:
                        logging.warning(f"Failed to create AtomicData for configuration: {str(e)}")
                        # Add the original config to the results without pseudolabels
                        updated_configs.append(copy.deepcopy(config))
                
                if not batch_inputs:
                    logging.warning(f"No valid inputs in batch {batch_idx+1}, skipping")
                    continue
                
                # Collate batch and move to device
                try:
                    from torch_geometric.data import Batch
                    collated_batch = Batch.from_data_list(batch_inputs).to(device)
                except Exception as e:
                    logging.error(f"Failed to collate batch: {str(e)}")
                    # Add the original configs to the results without pseudolabels
                    updated_configs.extend([copy.deepcopy(config) for config in valid_configs])
                    continue
                
                # Get model predictions
                try:
                    outputs = model(collated_batch)
                except Exception as e:
                    logging.error(f"Model forward pass failed: {str(e)}")
                    # Add the original configs to the results without pseudolabels
                    updated_configs.extend([copy.deepcopy(config) for config in valid_configs])
                    continue
                
                # Verify model outputs contain expected fields
                if "energy" not in outputs or "forces" not in outputs:
                    logging.error(f"Model outputs missing required fields. Available keys: {outputs.keys()}")
                    updated_configs.extend([copy.deepcopy(config) for config in valid_configs])
                    continue
                
                # Extract predictions
                energies = outputs["energy"].cpu().numpy()
                forces = outputs["forces"].cpu().numpy()
                
                # Split forces by structure
                forces_by_structure = []
                start_idx = 0
                for config in valid_configs:
                    num_atoms = len(config.atomic_numbers)
                    forces_by_structure.append(forces[start_idx:start_idx + num_atoms])
                    start_idx += num_atoms
                
                # Update configurations with pseudolabels
                for i, config in enumerate(valid_configs):
                    # Create a deep copy of the configuration
                    new_config = copy.deepcopy(config)
                    
                    # Check if the config uses a properties dictionary or direct attributes
                    has_properties_dict = hasattr(new_config, 'properties') and isinstance(new_config.properties, dict)
                    
                    # Store original values if they exist
                    if has_properties_dict:
                        # For configurations using properties dictionary
                        if 'energy' in new_config.properties:
                            if not hasattr(new_config, 'original_properties'):
                                new_config.original_properties = {}
                            new_config.original_properties['energy'] = new_config.properties['energy']
                        
                        if 'forces' in new_config.properties:
                            if not hasattr(new_config, 'original_properties'):
                                new_config.original_properties = {}
                            new_config.original_properties['forces'] = copy.deepcopy(new_config.properties['forces'])
                        
                        # Set new pseudolabels
                        new_config.properties['energy'] = float(energies[i])
                        new_config.properties['forces'] = forces_by_structure[i]
                    else:
                        # For configurations using direct attributes
                        if hasattr(new_config, 'energy') and new_config.energy is not None:
                            new_config.original_energy = new_config.energy
                        if hasattr(new_config, 'forces') and new_config.forces is not None:
                            new_config.original_forces = copy.deepcopy(new_config.forces)
                        
                        # Set new pseudolabels
                        new_config.energy = float(energies[i])
                        new_config.forces = forces_by_structure[i]
                    
                    updated_configs.append(new_config)
            
            except Exception as e:
                logging.error(f"Error in batch {batch_idx+1}: {str(e)}")
                logging.debug(traceback.format_exc())
                # Add the original configs to the results without pseudolabels
                updated_configs.extend([copy.deepcopy(config) for config in batch])
    
    # Ensure we have the same number of configurations as input
    if len(updated_configs) != len(configs):
        logging.warning(f"Number of output configurations ({len(updated_configs)}) doesn't match input ({len(configs)})")
        # Fill in any missing configs
        if len(updated_configs) < len(configs):
            config_ids = {id(config) for config in updated_configs}
            for config in configs:
                if id(config) not in config_ids:
                    updated_configs.append(copy.deepcopy(config))
    
    logging.info(f"Generated pseudolabels for {len(updated_configs)} configurations")
    return updated_configs


def apply_pseudolabels_to_pt_head_configs(
    foundation_model, 
    pt_head_config, 
    r_max, 
    device, 
    batch_size=32
):
    """
    Apply pseudolabels to pt_head configurations using the foundation model.
    
    Args:
        foundation_model: The pre-loaded foundation model
        pt_head_config: The HeadConfig object for pt_head
        r_max: Cutoff radius
        device: Device to run model on
        batch_size: Batch size for inference
        
    Returns:
        bool: True if pseudolabeling was successful, False otherwise
    """
    import logging
    
    if foundation_model is None:
        logging.warning("No foundation model provided. Skipping pseudolabeling.")
        return False
        
    if pt_head_config is None:
        logging.warning("No pt_head configuration found. Skipping pseudolabeling.")
        return False
        
    if not hasattr(pt_head_config, 'collections') or not pt_head_config.collections:
        logging.warning("No collections found in pt_head config. Skipping pseudolabeling.")
        return False
    
    try:
        logging.info("Applying pseudolabels to pt_head configurations using foundation model")
        
        # Move foundation model to the correct device
        foundation_model.to(device)
        
        # Use foundation model's z_table if available
        if hasattr(foundation_model, 'atomic_numbers'):
            from mace.tools.utils import AtomicNumberTable
            z_table = AtomicNumberTable(sorted(foundation_model.atomic_numbers.tolist()))
            logging.info(f"Using foundation model's atomic numbers for pseudolabeling: {z_table.zs}")
        elif hasattr(pt_head_config, 'z_table') and pt_head_config.z_table is not None:
            z_table = pt_head_config.z_table
            logging.info(f"Using pt_head's z_table for pseudolabeling: {z_table.zs}")
        else:
            logging.warning("No atomic number table available for pseudolabeling")
            return False
        
        # Process training configurations
        if hasattr(pt_head_config.collections, 'train') and pt_head_config.collections.train:
            logging.info(f"Generating pseudolabels for {len(pt_head_config.collections.train)} pt_head training configurations")
            updated_train_configs = generate_pseudolabels_for_configs(
                model=foundation_model,
                configs=pt_head_config.collections.train,
                z_table=z_table,
                r_max=r_max,
                device=device,
                batch_size=batch_size
            )
            
            # Replace the original configurations with updated ones
            pt_head_config.collections.train = updated_train_configs
            logging.info(f"Successfully applied pseudolabels to {len(updated_train_configs)} training configurations")
        
        # Process validation configurations if they exist
        if hasattr(pt_head_config.collections, 'valid') and pt_head_config.collections.valid:
            logging.info(f"Generating pseudolabels for {len(pt_head_config.collections.valid)} pt_head validation configurations")
            updated_valid_configs = generate_pseudolabels_for_configs(
                model=foundation_model,
                configs=pt_head_config.collections.valid,
                z_table=z_table,
                r_max=r_max,
                device=device,
                batch_size=batch_size
            )
            
            # Replace the original configurations with updated ones
            pt_head_config.collections.valid = updated_valid_configs
            logging.info(f"Successfully applied pseudolabels to {len(updated_valid_configs)} validation configurations")
        
        return True
    
    except Exception as e:
        logging.error(f"Error applying pseudolabels: {str(e)}")
        return False
