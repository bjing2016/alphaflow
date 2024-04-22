from alphaflow.utils.parsing import parse_train_args
args = parse_train_args()

from alphaflow.utils.logging import get_logger
logger = get_logger(__name__)
import torch, tqdm, os, wandb
import pandas as pd

from functools import partial
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from alphaflow.model.wrapper import ESMFoldWrapper, AlphaFoldWrapper
from openfold.utils.import_weights import import_jax_weights_

torch.set_float32_matmul_precision("high")
from alphaflow.config import model_config
from alphaflow.data.data_modules import OpenFoldSingleDataset, OpenFoldBatchCollator, OpenFoldDataset
from alphaflow.data.inference import CSVDataset, AlphaFoldCSVDataset

config = model_config(
    'initial_training',
    train=True, 
    low_prec=True
) 

loss_cfg = config.loss
data_cfg = config.data
data_cfg.common.use_templates = False
data_cfg.common.max_recycling_iters = 0

def load_clusters(path):
    cluster_size = []
    with open(args.pdb_clusters) as f:
        for line in f:
            names = line.split()
            for name in names:
                cluster_size.append({'name': name, 'cluster_size': len(names)})
    return pd.DataFrame(cluster_size).set_index('name')
    
def main():
    
    if args.wandb:
        wandb.init(
            entity=os.environ["WANDB_ENTITY"],
            settings=wandb.Settings(start_method="fork"),
            project="alphaflow",
            name=args.run_name,
            config=args,
        )

    logger.info("Loading the chains dataframe")
    pdb_chains = pd.read_csv(args.pdb_chains, index_col='name')

    if args.filter_chains:
        clusters = load_clusters(args.pdb_clusters)
        pdb_chains = pdb_chains.join(clusters)
        pdb_chains = pdb_chains[pdb_chains.release_date < args.train_cutoff]
    
    trainset = OpenFoldSingleDataset(
        data_dir = args.train_data_dir,
        alignment_dir = args.train_msa_dir,
        pdb_chains = pdb_chains,
        config = data_cfg,
        mode = 'train',
        subsample_pos = args.sample_train_confs,
        first_as_template = args.first_as_template,
    )
    if args.normal_validate:
        val_pdb_chains = pd.read_csv(args.val_csv, index_col='name')
        valset = OpenFoldSingleDataset(
            data_dir = args.train_data_dir,
            alignment_dir = args.train_msa_dir,
            pdb_chains = val_pdb_chains,
            config = data_cfg,
            mode = 'train',
            subsample_pos = args.sample_val_confs,
            num_confs = args.num_val_confs,
            first_as_template = args.first_as_template,
        )   
    else:
        valset = AlphaFoldCSVDataset(
            data_cfg,
            args.val_csv,
            mmcif_dir=args.mmcif_dir,
            data_dir=args.train_data_dir,
            msa_dir=args.val_msa_dir,
        )
    if args.filter_chains:
        trainset = OpenFoldDataset([trainset], [1.0], args.train_epoch_len)
    
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        collate_fn=OpenFoldBatchCollator(),
        num_workers=args.num_workers,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        collate_fn=OpenFoldBatchCollator(),
        num_workers=args.num_workers,
        shuffle=not args.filter_chains,
    )


    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=args.epochs,
        limit_train_batches=args.limit_batches or 1.0,
        limit_val_batches=args.limit_batches or 1.0,
        num_sanity_val_steps=0,
        enable_progress_bar=not args.wandb,
        gradient_clip_val=args.grad_clip,
        callbacks=[ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"], 
            save_top_k=-1,
            every_n_epochs=args.ckpt_freq,
        )],
        accumulate_grad_batches=args.accumulate_grad,
        check_val_every_n_epoch=args.val_freq,
        logger=False,
    )
    if args.mode == 'esmfold':
        model = ESMFoldWrapper(config, args)
        if args.ckpt is None:
            logger.info("Loading the model")
            path = "esmfold_3B_v1.pt"
            model_data = torch.load(path)
            model_state = model_data["model"]
            model.esmfold.load_state_dict(model_state, strict=False)
            logger.info("Model has been loaded")
            
            if not args.no_ema:
                model.ema = ExponentialMovingAverage(
                    model=model.esmfold, decay=config.ema.decay
                ) # need to initialize EMA this way at the beginning
    elif args.mode == 'alphafold':
        model = AlphaFoldWrapper(config, args)
        if args.ckpt is None:
            logger.info("Loading the model")
            import_jax_weights_(model.esmfold, 'params_model_1.npz', version='model_3')
            if not args.no_ema:
                model.ema = ExponentialMovingAverage(
                    model=model.model, decay=config.ema.decay
                ) # need to initialize EMA this way at the beginning
    
    if args.restore_weights_only:
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'], strict=False)
        args.ckpt = None
        if not args.no_ema:
            model.ema = ExponentialMovingAverage(
                model=model.model, decay=config.ema.decay
            ) # need to initialize EMA this way at the beginning
    
    
    if args.validate:
        trainer.validate(model, val_loader, ckpt_path=args.ckpt)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)
    
if __name__ == "__main__":
    main()