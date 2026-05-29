import argparse
from pathlib import Path
import os

# ============================================================================
# Proposed by Copilot: Configure PyTorch CUDA memory management to reduce
# fragmentation. This is particularly important for long-running evaluation
# loops to prevent OOM errors with progressive memory accumulation.
# ============================================================================
#os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from omegaconf import OmegaConf
import wandb
from src.models import model_registry
from src.datasets import get_dataloaders
from src.engine.loops import run_full_test_loop
from os.path import join


def test(args, cfg):

    train_loader, val_loader = get_dataloaders(**cfg.data)
    model = model_registry.get_model(**cfg.model)
    model.to(cfg.device)
    model.eval()

    metrics = run_full_test_loop(
        model,
        val_loader,
        output_dir=Path(cfg.log_dir),
        device=cfg.device,
        use_amp=cfg.use_amp,
        **cfg.get('test_cfg', {}),
        include_full_ddf=args.include_full_ddf_metrics,
        save_predictions=args.save_predictions,
        save_images_with_predictions=args.save_predictions,
        images_key_for_save='raw_images',
        nr_scans=args.nr_scans,
    )
    print(metrics)
    
    if args.log_wandb: 
        wandb.init(
            project=cfg.logger_kw.wandb_project, 
            job_type='test', 
            config=OmegaConf.to_object(cfg), 
        )
        wandb.log({
            f'{k}/val': v for k, v in metrics.items()
        })


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('--train_dir')
    parser.add_argument('--config', '-c')
    parser.add_argument('--checkpoint', help='Checkpoint to load')
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--include_full_ddf_metrics', action='store_true')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--nr_scans', type=int, default=None)

    args = parser.parse_args()

    if args.train_dir:
        if args.log_dir is None: 
            args.log_dir = join(args.train_dir, 'test')
        if args.checkpoint is None: 
            args.checkpoint = join(args.train_dir, 'checkpoint', 'best.pt')
        if args.config is None: 
            args.config = join(args.train_dir, 'config_resolved.yaml')

    config = OmegaConf.load(args.config)
    if args.log_dir: 
        config.log_dir = args.log_dir
    if args.checkpoint: 
        config.model.checkpoint = args.checkpoint
    if args.nr_scans is not None:
        config.nr_scans = args.nr_scans

    test(args, config)
    