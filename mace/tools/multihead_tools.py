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
            
            # Extract the relevant outputs for the specific head
            # The model may return either single values (for single head models)
            # or per-head values (for multi-head models)
            
            # Handle energy - could be [batch_size] or [batch_size, num_heads]
            energy = out['energy'].cpu().detach() if 'energy' in out else None
            
            # Handle forces - format is typically [num_atoms, 3]
            forces = out['forces'].cpu().detach() if 'forces' in out else None
            
            # Handle stress - could be [batch_size, 3, 3] or [batch_size, num_heads, 3, 3]
            stress = None
            if 'stress' in out and out['stress'] is not None:
                stress = out['stress'].cpu().detach()
            
            # Handle virials - could be [batch_size, 3, 3] or [batch_size, num_heads, 3, 3]
            virials = None
            if 'virials' in out and out['virials'] is not None:
                virials = out['virials'].cpu().detach()
            
            # Handle dipole - could be [batch_size, 3] or [batch_size, num_heads, 3]
            dipole = None
            if 'dipole' in out and out['dipole'] is not None:
                dipole = out['dipole'].cpu().detach()
            
            # Handle charges - could be [num_atoms] or [num_heads, num_atoms]
            charges = None
            if 'charges' in out and out['charges'] is not None:
                charges = out['charges'].cpu().detach()
            
            # Create dictionary with the pseudolabels
            pseudo = {
                'energy': energy,
                'forces': forces,
                'stress': stress,
                'virials': virials,
                'dipole': dipole,
                'charges': charges
            }
            
            pseudolabels.append(pseudo)
            
        except RuntimeError as e:
            logging.error(f"Error generating pseudolabels: {str(e)}")
            
        # Restore the original head values if they were changed
        if 'head' in batch_dict and 'original_head' in locals():
            batch_dict['head'] = original_head
    
    # Restore original requires_grad settings for model parameters
    for name, param in model.named_parameters():
        if name in original_requires_grad:
            param.requires_grad_(original_requires_grad[name])
    
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
    
    # Debug log shapes of first item
    if len(dataset) > 0 and len(pseudolabels) > 0:
        data = dataset[0]
        pseudo = pseudolabels[0]
        logging.info("=== Debugging tensor shapes for first item ===")
        if hasattr(data, 'energy') and data.energy is not None:
            logging.info(f"Original energy shape: {data.energy.shape}")
        if 'energy' in pseudo and pseudo['energy'] is not None:
            logging.info(f"Pseudo energy shape: {pseudo['energy'].shape}")
        if hasattr(data, 'forces') and data.forces is not None:
            logging.info(f"Original forces shape: {data.forces.shape}")
        if 'forces' in pseudo and pseudo['forces'] is not None:
            logging.info(f"Pseudo forces shape: {pseudo['forces'].shape}")
    
    pseudolabeled_count = 0
    for i, (data, pseudo) in enumerate(zip(dataset, pseudolabels)):
        try:
            # Apply energy pseudolabel
            if 'energy' in pseudo and pseudo['energy'] is not None:
                # Handle case where energy is [batch_size, num_heads]
                if pseudo['energy'].dim() > 0:
                    if pseudo['energy'].dim() > 1:
                        # For multi-head output, use the first dimension (typically batch_size=1)
                        data.energy = pseudo['energy'][0]
                    else:
                        # For single value per batch item
                        data.energy = pseudo['energy'][0] if pseudo['energy'].numel() > 1 else pseudo['energy']
                else:
                    data.energy = pseudo['energy']
            
            # Apply forces pseudolabel
            if 'forces' in pseudo and pseudo['forces'] is not None:
                # Forces should be [num_atoms, 3]
                if data.forces is None or data.forces.shape != pseudo['forces'].shape:
                    logging.debug(f"Forces shape adjustment at index {i}. Original: {data.forces.shape if data.forces is not None else None}, Pseudo: {pseudo['forces'].shape}")
                data.forces = pseudo['forces']
            
            # Apply stress pseudolabel
            if 'stress' in pseudo and pseudo['stress'] is not None and hasattr(data, 'stress'):
                # Stress should be [1, 3, 3] for AtomicData
                stress = pseudo['stress']
                
                # Handle multi-head stress: [batch, head, 3, 3] -> [1, 3, 3]
                if stress.dim() == 4:
                    stress = stress[0, 0].unsqueeze(0)  # Take first batch, first head
                # Handle [batch, 3, 3] -> [1, 3, 3]
                elif stress.dim() == 3:
                    stress = stress[0].unsqueeze(0)  # Take first batch
                # Handle [3, 3] -> [1, 3, 3]
                elif stress.dim() == 2:
                    stress = stress.unsqueeze(0)
                
                data.stress = stress
            
            # Apply virials pseudolabel
            if 'virials' in pseudo and pseudo['virials'] is not None and hasattr(data, 'virials'):
                # Virials should be [1, 3, 3] for AtomicData
                virials = pseudo['virials']
                
                # Handle multi-head virials: [batch, head, 3, 3] -> [1, 3, 3]
                if virials.dim() == 4:
                    virials = virials[0, 0].unsqueeze(0)  # Take first batch, first head
                # Handle [batch, 3, 3] -> [1, 3, 3]
                elif virials.dim() == 3:
                    virials = virials[0].unsqueeze(0)  # Take first batch
                # Handle [3, 3] -> [1, 3, 3]
                elif virials.dim() == 2:
                    virials = virials.unsqueeze(0)
                
                data.virials = virials
            
            # Apply dipole pseudolabel
            if 'dipole' in pseudo and pseudo['dipole'] is not None and hasattr(data, 'dipole'):
                # Dipole should be [1, 3] for AtomicData
                dipole = pseudo['dipole']
                
                # Handle multi-head dipole: [batch, head, 3] -> [1, 3]
                if dipole.dim() == 3:
                    dipole = dipole[0, 0].unsqueeze(0)  # Take first batch, first head
                # Handle [batch, 3] -> [1, 3]
                elif dipole.dim() == 2:
                    dipole = dipole[0].unsqueeze(0)  # Take first batch
                # Handle [3] -> [1, 3]
                elif dipole.dim() == 1:
                    dipole = dipole.unsqueeze(0)
                
                data.dipole = dipole
            
            # Apply charges pseudolabel
            if 'charges' in pseudo and pseudo['charges'] is not None and hasattr(data, 'charges'):
                # Charges should be [num_atoms] for AtomicData
                charges = pseudo['charges']
                
                # Handle multi-head charges: [head, num_atoms] -> [num_atoms]
                if charges.dim() == 2:
                    charges = charges[0]  # Take first head
                
                data.charges = charges
            
            pseudolabeled_count += 1
            
        except Exception as e:
            logging.error(f"Error applying pseudolabels at index {i}: {str(e)}")
            # Continue with the next sample rather than failing completely
            continue
    
    logging.info(f"Successfully applied pseudolabels to {pseudolabeled_count}/{len(dataset)} data points")
    return dataset
