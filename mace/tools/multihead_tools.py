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
    """Generate pseudolabels using the foundation model for multihead models."""
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
        
        # Store original head values
        original_head = None
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
            
            # Extract original samples from the batch
            for i in range(len(batch)):
                # Get the original sample with all its data
                original_sample = batch.get_example(i)
                
                # Create a new dictionary with all fields from the original sample
                sample_dict = original_sample.to_dict()
                
                # Update the fields with pseudolabels from model output
                # Energy
                if 'energy' in out and out['energy'] is not None:
                    energy_tensor = out['energy'].cpu().detach()
                    if energy_tensor.dim() > 0 and i < energy_tensor.size(0):
                        sample_dict['energy'] = energy_tensor[i].view([])
                
                # Forces
                if 'forces' in out and out['forces'] is not None:
                    forces_tensor = out['forces'].cpu().detach()
                    atoms_indices = (batch.batch == i).cpu()
                    if atoms_indices.any():
                        forces = forces_tensor[atoms_indices].clone()
                        if forces.shape[0] == sample_dict['positions'].shape[0]:
                            sample_dict['forces'] = forces
                
                # Stress - for multihead models only
                if 'stress' in out and out['stress'] is not None:
                    stress_tensor = out['stress'].cpu().detach()
                    if stress_tensor.dim() == 4 and i < stress_tensor.size(0):  # [batch, heads, 3, 3]
                        stress = stress_tensor[i, head_idx].clone()
                        if stress.shape == (3, 3):
                            sample_dict['stress'] = stress.unsqueeze(0)
                
                # Virials - for multihead models only
                if 'virials' in out and out['virials'] is not None:
                    virials_tensor = out['virials'].cpu().detach()
                    if virials_tensor.dim() == 4 and i < virials_tensor.size(0):  # [batch, heads, 3, 3]
                        virials = virials_tensor[i, head_idx].clone()
                        if virials.shape == (3, 3):
                            sample_dict['virials'] = virials.unsqueeze(0)
                
                # Dipole - for multihead models only
                if 'dipole' in out and out['dipole'] is not None:
                    dipole_tensor = out['dipole'].cpu().detach()
                    if dipole_tensor.dim() == 3 and i < dipole_tensor.size(0):  # [batch, heads, 3]
                        dipole = dipole_tensor[i, head_idx].clone()
                        if dipole.shape == (3,):
                            sample_dict['dipole'] = dipole.unsqueeze(0)
                
                # Charges - for multihead models only
                if 'charges' in out and out['charges'] is not None:
                    charges_tensor = out['charges'].cpu().detach()
                    atoms_indices = (batch.batch == i).cpu()
                    if atoms_indices.any() and charges_tensor.dim() == 2:  # [heads, total_atoms]
                        charges = charges_tensor[head_idx, atoms_indices].clone()
                        if charges.shape[0] == sample_dict['positions'].shape[0]:
                            sample_dict['charges'] = charges
                
                # Restore the original head value if it was changed
                if original_head is not None:
                    sample_dict['head'] = original_head[i].clone()
                
                # Create a new AtomicData object from the dictionary
                from mace.data import AtomicData
                pseudo_data = AtomicData.from_dict(sample_dict)
                
                # Append the complete AtomicData object
                pseudolabels.append(pseudo_data)
                
        except RuntimeError as e:
            logging.error(f"Error generating pseudolabels: {str(e)}")
            continue
        
    # Restore original requires_grad settings for model parameters
    for name, param in model.named_parameters():
        if name in original_requires_grad:
            param.requires_grad_(original_requires_grad[name])
    
    logging.info(f"Generated {len(pseudolabels)} pseudolabels from {len(data_loader)} batches")
    return pseudolabels

def apply_pseudolabels(dataset, pseudolabels):
    """Replace original values with pseudolabels in the dataset."""
    if len(pseudolabels) == 0:
        logging.warning("No pseudolabels generated. Continuing with original dataset.")
        return dataset
        
    if len(dataset) != len(pseudolabels):
        logging.warning(f"Dataset length ({len(dataset)}) does not match pseudolabels length ({len(pseudolabels)}). "
                       f"This may indicate a problem with batch processing.")
        # Truncate to the shorter length to avoid errors
        min_length = min(len(dataset), len(pseudolabels))
        dataset = dataset[:min_length]
        pseudolabels = pseudolabels[:min_length]
    
    # Log some information about the pseudolabeled data
    if len(dataset) > 0 and len(pseudolabels) > 0:
        try:
            # Verify that pseudolabels have all required fields
            first_pseudo = pseudolabels[0]
            first_orig = dataset[0]
            
            # Log the keys to help with debugging
            pseudo_keys = sorted(first_pseudo.keys)
            orig_keys = sorted(first_orig.keys)
            
            if set(orig_keys) != set(pseudo_keys):
                missing_keys = set(orig_keys) - set(pseudo_keys)
                extra_keys = set(pseudo_keys) - set(orig_keys)
                
                if missing_keys:
                    logging.error(f"Pseudolabels missing required keys: {missing_keys}")
                
                if extra_keys:
                    logging.warning(f"Pseudolabels contain extra keys: {extra_keys}")
                
                # If there are missing keys, we'll use the original dataset
                if missing_keys:
                    logging.warning("Using original dataset due to missing keys in pseudolabels")
                    return dataset
            
            logging.info(f"Replacing dataset with {len(pseudolabels)} pseudolabeled samples")
            return pseudolabels
            
        except Exception as e:
            logging.error(f"Error validating pseudolabels: {str(e)}")
            logging.warning("Using original dataset due to error in pseudolabels")
            return dataset
    
    return dataset
