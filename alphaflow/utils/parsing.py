from argparse import ArgumentParser
import subprocess, os


def parse_train_args():
    parser = ArgumentParser()

    parser.add_argument("--mode", choices=['esmfold', 'alphafold'], default='alphafold')
    
    ## Trainer settings
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--restore_weights_only", action='store_true')
    parser.add_argument("--validate", action='store_true', default=False)
    
    ## Epoch settings
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_epoch_len", type=int, default=40000)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)

    ## Optimization settings
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--check_grad", action="store_true")
    parser.add_argument("--accumulate_grad", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_ema", action='store_true')
    
    ## Training data 
    parser.add_argument("--train_data_dir", type=str, default='./data')
    parser.add_argument("--pdb_chains", type=str, default='./pdb_chains_msa.csv')
    parser.add_argument("--train_msa_dir", type=str, default='./msa_dir')
    parser.add_argument("--pdb_clusters", type=str, default='./pdb_clusters')
    parser.add_argument("--train_cutoff", type=str, default='2021-10-01')
    parser.add_argument("--mmcif_dir", type=str, default='./mmcif_dir')
    parser.add_argument("--filter_chains", action='store_true')
    parser.add_argument("--sample_train_confs", action='store_true')
    
    ## Validation data
    parser.add_argument("--val_csv", type=str, default='splits/cameo2022.csv')
    parser.add_argument("--val_samples", type=int, default=5)
    parser.add_argument("--val_msa_dir", type=str, default='./alignment_dir')
    parser.add_argument("--sample_val_confs", action='store_true')
    parser.add_argument("--num_val_confs", type=int, default=None)
    parser.add_argument("--normal_validate", action='store_true')
    
    ## Flow matching
    parser.add_argument("--noise_prob", type=float, default=0.5)
    parser.add_argument("--self_cond_prob", type=float, default=0.5)
    parser.add_argument("--extra_input", action='store_true')
    parser.add_argument("--extra_input_prob", type=float, default=0.5)
    parser.add_argument("--first_as_template", action='store_true')
    parser.add_argument("--distillation", action='store_true')
    parser.add_argument("--distill_self_cond", action='store_true')
    
    ## Logging args
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--ckpt_freq", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="default")
    
    args = parser.parse_args()
    os.environ["MODEL_DIR"] = os.path.join("workdir", args.run_name)
    os.environ["WANDB_LOGGING"] = str(int(args.wandb))
    if args.wandb:
        if subprocess.check_output(["git", "status", "-s"]):
            exit()
    args.commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )

    return args
    