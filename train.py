import warnings
warnings.filterwarnings('ignore')

import gc
import numpy as np
import os
import wandb
import argparse
import torch
import random
import utils
from model.model import ESM_model, AF2_model, BaselineOHE
from data.dataloader import DynaData
from torch.utils.data.dataloader import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(args, model, dloader, optimizer=None, step='train', method='esm2', accum_steps=1):
    """Train the model for one epoch."""

    total_loss, total_mask = 0, 0
    all_labels, all_logits, all_mask =[], [], []
    
    for i, batch in enumerate(dloader):
        
        targets = batch['targets'].to(DEVICE,non_blocking=True)
        eval_mask = batch['eval_mask'].to(DEVICE,non_blocking=True)
        seq_id =  batch['seq_id'].to(DEVICE,non_blocking=True).bool()        
        
        # Forward pass
        if 'esm3' in method:
            seq_input, struct_input = None, None
            if args.seq:
                seq_input = batch['seqs'].to(DEVICE,non_blocking=True)
            if args.struct:
                struct_input = batch['structs'].to(DEVICE,non_blocking=True)
            logits = model((seq_input, struct_input), seq_id)
        elif 'esm2' in args.method:
            seq_input = batch['seqs'].to(DEVICE,non_blocking=True)
            logits = model(seq_input, seq_id)
        elif 'esmc' in args.method:
            seq_input = batch['seqs'].to(DEVICE,non_blocking=True)
            logits = model(seq_input, seq_id)
        elif 'af2' in args.method:
            pair_reps = batch['pair_reps'].to(DEVICE, non_blocking=True).to(torch.float32)
            logits = model(pair_reps, seq_id) 
        elif 'baseline' in args.method:
            seq_input = batch['seqs'].to(DEVICE,non_blocking=True)
            logits = model(seq_input, seq_id)
      
        # Compute loss with label smoothing
        loss, mask_, mean_loss = utils.get_loss(logits, targets, eval_mask, label_smoothing=args.label_smoothing)

        total_loss += loss.item()
        total_mask += mask_.item()
        
        # Backward pass and optimization (with gradient accumulation)
        if step == 'train':
            mean_loss.backward()
            if (i + 1) % accum_steps == 0 or (i + 1) == len(dloader):
                optimizer.step()
                optimizer.zero_grad()
        elif step == 'eval':
            all_labels.extend(targets)
            all_logits.extend(logits)
            all_mask.extend(eval_mask)
        
        if step == 'train' and i >= 0 and i % 50 == 0:
            masked_logits, masked_labels = utils.get_masked(logits, targets, eval_mask)
            train_auroc, _, train_nauprc = utils.get_auroc(masked_logits, masked_labels)
            if args.debugging:
                print(f"{step} loss: {mean_loss.item()}, {step} AUROC: {train_auroc}, {step} normAUPRC: {train_nauprc}")
            else:
                wandb.log({ f"{step}_loss": mean_loss.item(), f"{step}_rel_auroc": train_auroc, f"{step}_rel_nauprc": train_nauprc})
    
    if step == 'train':     
        return total_loss / total_mask
    final_loss = total_loss / total_mask
    
    masked_logits, masked_labels = utils.get_masked(torch.cat(all_logits), 
                                                    torch.cat(all_labels), 
                                                    torch.cat(all_mask))
    final_auroc, _, final_nauprc = utils.get_auroc(masked_logits, masked_labels)
    
    if not args.debugging:
        wandb.log({ f"{step}_loss": final_loss, 
                    f"{step}_auroc": final_auroc,
                    f"{step}_nauprc": final_nauprc})
    print(f"{step} loss: {final_loss}. {step} AUROC: {final_auroc}. {step} nAUPRC: {final_nauprc}")

    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_loss, final_auroc


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load a model checkpoint and optimizer"""

    if not os.path.isfile(checkpoint_path):
        exit('Model checkpoint does not exist.')

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch

def main(args):
    
    config, config_dict = utils.load_config(f'configs/{args.method}.yml', return_dict=True)

    print(f'Running on {DEVICE}, an {torch.cuda.get_device_name(0)}')
    
    # Set path of model weights
    if 'esm2' in args.method or 'esm3' in args.method or 'esmc' in args.method:
        save_path = os.path.join(args.save_dir, f'layer{args.layer}.pt')
    else:
        save_path = os.path.join(args.save_dir, f'{args.method}.pt')
    
    # Set up dict for WandB logging
    config_dict['layer'] = args.layer
    config_dict['seq'] = args.seq
    config_dict['struct'] = args.struct
    config_dict['dir']['save_dir'] = args.save_dir
    config_dict['seed'] = args.seed
    config_dict['dir']['save_path'] = save_path

    # WandB logging
    if not args.debugging:
        wandb.init(
            settings=wandb.Settings(_service_wait=300),
            entity= config.wandb.team,
            project = config.wandb.project,
            dir = config.wandb.dir,
            config = config_dict
        )
    
    # Initialize the model
    if 'esm' in args.method:
        model = ESM_model(args.method,
                          nlayers = config.model.nlayers,
                          nheads = config.model.nheads,
                          dropout = config.train.dropout,
                          layer = args.layer).to(DEVICE,non_blocking=True)
    elif 'af2' == args.method:
        model = AF2_model(res_count = config.model.res_count, 
                          hidden_size = config.model.hidden_size,
                          length = config.model.length,
                          nlayers = config.model.nlayers,
                          nheads = config.model.nheads,
                          dropout = config.train.dropout).to(DEVICE,non_blocking=True).to(torch.float32)
    elif 'baseline' in args.method:
        model = BaselineOHE(nlayers = config.model.nlayers,
                            nheads = config.model.nheads,
                            dropout = config.train.dropout).to(DEVICE,non_blocking=True)
   
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr= config.train.lr)

    ## Create datasets and dataloader
    # Training set
    train = DynaData(config.data.train.split,
                    type = config.data.train.type,
                    crop_len = config.data.train.crop_len, 
                    cluster_file = config.data.cluster,
                    sample_clusters = config.data.sample_clusters, 
                    pair_rep = config.data.pair_rep,
                    method = (args.method, model))
    train_loader = DataLoader(dataset = train, 
                              batch_size = config.train.batchsize,
                              collate_fn = train.__collate_fn__, 
                              shuffle = True, 
                              drop_last = True, 
                              num_workers = config.train.num_workers,
                              pin_memory = True)
    # Validation set
    val = DynaData(config.data.val.split,
                  type = config.data.val.type,
                  crop_len = config.data.val.crop_len, 
                  cluster_file = config.data.cluster,
                  sample_clusters = config.data.sample_clusters, 
                  pair_rep = config.data.pair_rep,
                  method = (args.method, model))
    val_loader = DataLoader(dataset = val, 
                            batch_size = config.train.batchsize,
                            collate_fn = val.__collate_fn__, 
                            shuffle = True, 
                            drop_last = True,
                            num_workers = config.train.num_workers, 
                            pin_memory = True)

    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()

    # Continue training if model has been trained before
    if os.path.exists(save_path):
        model, optimizer, init_epoch = load_checkpoint(model, optimizer, save_path)
        with torch.no_grad():
            _, auroc_val = run(args, model, val_loader, step='eval', method=args.method)
        best_auroc = auroc_val
    else:
        init_epoch, best_auroc, auroc_val = 0,0,0

    # Begin training  
    for epoch in range(init_epoch, config.train.epochs):
        print(f'epoch:{epoch + 1} start!', args.method)
        
        # Train
        model.train()
        
        run(args, model, train_loader, 
            step = 'train',
            optimizer = optimizer,
            method = args.method,
            accum_steps = config.train.accum_steps)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            _, auroc_val = run(args, model, val_loader, step='eval', method=args.method)
            if auroc_val > best_auroc:
                best_auroc = auroc_val
                if not args.debugging:
                    # Save model weights
                    save_data = {"model_state_dict": model.state_dict(), 
                                 "optim_state_dict": optimizer.state_dict(),
                                 "epoch": epoch, 
                                 "layer": args.layer}
                    torch.save(save_data, save_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='ProtXM Training Script')
    parser.add_argument('--method', type=str, default='esm2', choices=['esm2', 'esm3', 'esmc', 'af2', 'baseline'], help='Model type')
    parser.add_argument('--layer', default = -1, type=int)
    parser.add_argument('--slurm_job_id', type=str, help='slurm_job_id')
    parser.add_argument('--debugging', action='store_true')
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--seq', action='store_true')
    parser.add_argument('--struct', action='store_true')
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor (0.0 to 1.0)')
    
    args = parser.args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.method == 'esm3' and not (args.struct or args.seq):
        exit('ESM3 requires at least one input of seq or struct. Exiting now...')
    if args.method == 'esm3' and (args.layer < 0 or args.layer > 47):
        exit('ESM3 only has 48 layers, accepted values are [0-47]')
    if (args.method == 'esm2' or args.method == 'esmc') and (args.layer < 0 or args.layer > 30):
        exit('ESM2 and ESMC have only has 31 layers, accepted values are [0-30]')
    if args.label_smoothing < 0.0 or args.label_smoothing > 1.0:
        exit('Label smoothing must be between 0.0 and 1.0')
    main(args)
