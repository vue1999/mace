import argparse
import ast
import dataclasses
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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


def get_pseudolabels(model, data_loader, device):
    """Generate pseudolabels using the foundation model."""
    model.eval()
    pseudolabels = []
    
    # Disable gradient tracking for model parameters while keeping gradients for input tensors
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
    
    # Get the index of pt_head if available
    try:
        head_idx = model.heads.index("pt_head")
        logging.info(f"Found 'pt_head' at index {head_idx} for pseudolabeling")
    except (ValueError, AttributeError):
        head_idx = 0
        logging.warning("Could not find 'pt_head', using index 0 instead for pseudolabeling")
    
    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        
        # Set the head to pt_head for all batch items to ensure consistent output
        if 'head' in batch_dict:
            original_head = batch_dict['head'].clone()
            batch_dict['head'] = torch.full_like(batch_dict['head'], head_idx)
        
        try:
            # Run the model with forces computation enabled
            out = model(
                batch_dict,
                training=False,
                compute_force=True,
                compute_virials=True,
                compute_stress=True
            )
            
            # -------------------------------------------------------------
            # Unbatch the model outputs to create per-sample pseudolabels
            # -------------------------------------------------------------
            
            # Now unbatch the results - we need to split batch outputs into individual samples
            for i in range(len(batch)):
                sample_data = batch.get_example(i)
                sample_idx = i
                
                # Create a copy of the original data to modify with pseudolabels
                # This ensures we maintain all the required fields for AtomicData
                pseudo_data = sample_data.clone()
                
                # Extract energy - scalar per sample
                if 'energy' in out and out['energy'] is not None:
                    energy_tensor = out['energy'].cpu().detach()
                    if energy_tensor.dim() > 0:
                        if sample_idx < energy_tensor.size(0):
                            energy = energy_tensor[sample_idx].clone()
                            # Ensure it's a scalar (0-dimensional tensor)
                            pseudo_data.energy = energy.view([])
                
                # Extract forces - need to extract atoms for this specific sample
                if 'forces' in out and out['forces'] is not None:
                    forces_tensor = out['forces'].cpu().detach()
                    # Extract atoms for this sample using batch.get_example
                    atoms_indices = (batch.batch == sample_idx).cpu()
                    if atoms_indices.any():
                        forces = forces_tensor[atoms_indices].clone()
                        if forces.shape[0] == pseudo_data.positions.shape[0]:
                            pseudo_data.forces = forces
                
                # Extract stress
                if 'stress' in out and out['stress'] is not None:
                    stress_tensor = out['stress'].cpu().detach()
                    if stress_tensor.dim() == 3:  # [batch, 3, 3]
                        if sample_idx < stress_tensor.size(0):
                            stress = stress_tensor[sample_idx].clone()
                            # Make sure it's [1, 3, 3]
                            if stress.shape == (3, 3):
                                stress = stress.unsqueeze(0)
                            pseudo_data.stress = stress
                    elif stress_tensor.dim() == 4:  # [batch, heads, 3, 3] 
                        if sample_idx < stress_tensor.size(0):
                            stress = stress_tensor[sample_idx, head_idx].clone()
                            # Make sure it's [1, 3, 3]
                            if stress.shape == (3, 3):
                                stress = stress.unsqueeze(0)
                            pseudo_data.stress = stress
                
                # Extract virials
                if 'virials' in out and out['virials'] is not None:
                    virials_tensor = out['virials'].cpu().detach()
                    if virials_tensor.dim() == 3:  # [batch, 3, 3]
                        if sample_idx < virials_tensor.size(0):
                            virials = virials_tensor[sample_idx].clone()
                            # Make sure it's [1, 3, 3]
                            if virials.shape == (3, 3):
                                virials = virials.unsqueeze(0)
                            pseudo_data.virials = virials
                    elif virials_tensor.dim() == 4:  # [batch, heads, 3, 3]
                        if sample_idx < virials_tensor.size(0):
                            virials = virials_tensor[sample_idx, head_idx].clone()
                            # Make sure it's [1, 3, 3]
                            if virials.shape == (3, 3):
                                virials = virials.unsqueeze(0)
                            pseudo_data.virials = virials
                
                # Extract dipole
                if 'dipole' in out and out['dipole'] is not None:
                    dipole_tensor = out['dipole'].cpu().detach()
                    if dipole_tensor.dim() == 2:  # [batch, 3]
                        if sample_idx < dipole_tensor.size(0):
                            dipole = dipole_tensor[sample_idx].clone()
                            # Make sure it's [1, 3]
                            if dipole.shape == (3,):
                                dipole = dipole.unsqueeze(0)
                            pseudo_data.dipole = dipole
                    elif dipole_tensor.dim() == 3:  # [batch, heads, 3]
                        if sample_idx < dipole_tensor.size(0):
                            dipole = dipole_tensor[sample_idx, head_idx].clone()
                            # Make sure it's [1, 3]
                            if dipole.shape == (3,):
                                dipole = dipole.unsqueeze(0)
                            pseudo_data.dipole = dipole
                
                # Extract charges
                if 'charges' in out and out['charges'] is not None:
                    charges_tensor = out['charges'].cpu().detach()
                    # Extract atoms for this sample
                    atoms_indices = (batch.batch == sample_idx).cpu()
                    if atoms_indices.any():
                        if charges_tensor.dim() == 1:  # [total_atoms]
                            charges = charges_tensor[atoms_indices].clone()
                            if charges.shape[0] == pseudo_data.positions.shape[0]:
                                pseudo_data.charges = charges
                        elif charges_tensor.dim() == 2:  # [heads, total_atoms]
                            charges = charges_tensor[head_idx, atoms_indices].clone()
                            if charges.shape[0] == pseudo_data.positions.shape[0]:
                                pseudo_data.charges = charges
                
                # Append the complete AtomicData object with pseudolabels
                pseudolabels.append(pseudo_data)
                
        except RuntimeError as e:
            logging.error(f"Error generating pseudolabels: {str(e)}")
        
        # Restore the original head values if they were changed
        if 'head' in batch_dict and 'original_head' in locals():
            batch_dict['head'] = original_head
    
    # Restore original requires_grad settings for model parameters
    for name, param in model.named_parameters():
        if name in original_requires_grad:
            param.requires_grad_(original_requires_grad[name])
    
    logging.info(f"Generated {len(pseudolabels)} pseudolabels from {len(data_loader)} batches")
    return pseudolabels

def apply_pseudolabels(dataset, pseudolabels):
    """Replace original values with pseudolabels in the dataset."""
    if len(dataset) != len(pseudolabels):
        logging.warning(f"Dataset length ({len(dataset)}) does not match pseudolabels length ({len(pseudolabels)}). "
                       f"This may indicate a problem with batch processing.")
        # Truncate to the shorter length to avoid errors
        min_length = min(len(dataset), len(pseudolabels))
        dataset = dataset[:min_length]
        pseudolabels = pseudolabels[:min_length]
    
    # Log some information about the pseudolabeled data
    if len(dataset) > 0 and len(pseudolabels) > 0:
        logging.info(f"Replacing dataset with {len(pseudolabels)} pseudolabeled samples")
        
    # Since we now return complete AtomicData objects, we can just replace the dataset
    return pseudolabels
