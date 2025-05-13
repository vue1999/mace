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
    """Generate pseudolabels using the foundation model for multihead models.
    Returns a list of modified data samples with pseudolabels.
    """
    model.eval()
    all_batches_with_pseudolabels = []
    
    # Disable gradient tracking for model parameters, but keep gradient tracking
    # for positional inputs to allow force/stress calculation
    original_requires_grad = {}
    for param in model.parameters():
        original_requires_grad[param] = param.requires_grad
        param.requires_grad = False
    
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
        
        try:
            # Run the model with forces, virials, and stress computation enabled
            out = model(
                batch_dict,
                training=False,
                compute_force=True,
                compute_virials=True,
                compute_stress=True
            )
            
            # Create a copy of batch to add pseudolabels
            logging.info(f"Batch: {batch}")
            batch_with_labels = batch.clone()
            logging.info(f"Batch with labels: {batch_with_labels}")
            
            # Energy pseudolabel (per graph)
            if 'energy' in out and out['energy'] is not None:
                batch_with_labels.energy = out['energy']
            
            # Forces pseudolabel (per atom)
            if 'forces' in out and out['forces'] is not None:
                batch_with_labels.forces = out['forces']
            
            # Stress pseudolabel (per graph)
            if 'stress' in out and out['stress'] is not None:
                # Select the specific head's stress predictions
                if out['stress'].dim() == 4:  # [batch, heads, 3, 3]
                    batch_with_labels.stress = out['stress'][:, head_idx]
                else:
                    batch_with_labels.stress = out['stress']
            
            # Virials pseudolabel (per graph)
            if 'virials' in out and out['virials'] is not None:
                # Select the specific head's virials predictions
                if out['virials'].dim() == 4:  # [batch, heads, 3, 3]
                    batch_with_labels.virials = out['virials'][:, head_idx]
                else:
                    batch_with_labels.virials = out['virials']
            
            # Dipole pseudolabel (per graph)
            if 'dipole' in out and out['dipole'] is not None:
                # Select the specific head's dipole predictions
                if out['dipole'].dim() == 3:  # [batch, heads, 3]
                    batch_with_labels.dipole = out['dipole'][:, head_idx]
                else:
                    batch_with_labels.dipole = out['dipole']
            
            # Charges pseudolabel (per atom)
            if 'charges' in out and out['charges'] is not None:
                # Select the specific head's charges predictions
                if out['charges'].dim() == 2:  # [heads, total_atoms]
                    batch_with_labels.charges = out['charges'][head_idx]
                else:
                    batch_with_labels.charges = out['charges']
            
            # Add this batch to our collection
            all_batches_with_pseudolabels.append(batch_with_labels.to("cpu"))
                
        except RuntimeError as e:
            logging.error(f"Error generating pseudolabels: {str(e)}")
            continue
    
    # Restore original requires_grad settings for model parameters
    for param, requires_grad in original_requires_grad.items():
        param.requires_grad = requires_grad
    
    # Convert all batches to a list of data objects
    samples_with_pseudolabels = []
    for batch in all_batches_with_pseudolabels:
        samples_with_pseudolabels.extend(batch.to_data_list())
    
    logging.info(f"Generated pseudolabels for {len(samples_with_pseudolabels)} samples")
    return samples_with_pseudolabels

def apply_pseudolabels(dataset, pseudolabels):
    """Replace the dataset with pseudolabeled data."""
    if len(pseudolabels) == 0:
        logging.warning("No pseudolabels generated. Continuing with original dataset.")
        return dataset
    
    # Simply return the pseudolabeled dataset
    logging.info(f"Applied pseudolabels to {len(pseudolabels)} samples")
    return pseudolabels
