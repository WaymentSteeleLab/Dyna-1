import warnings
warnings.filterwarnings("ignore")
from collections import OrderedDict
import re
import utils
import torch
import random
import argparse
import numpy as np
import pandas as pd
import MDAnalysis as mda

from model.model import *
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def handle_name(args):
    """Processes the output file name given inputs of args.name and args.pdb; otherwise generates a random number"""
    if args.name:
        pdb_name = args.name
    elif args.pdb:
        if len(args.pdb) == 4:
            pdb_name = args.pdb
        else:
            pdb_name = args.pdb.split('/') [-1][:-4]
    else:
        pdb_name = random.randint(0, 100000)
    return f'{pdb_name}-Dyna1'

def get_pdb_from_upload(args, pdb_id):
    fixed_pdb = None
    try:
        protein_chain = ProteinChain.from_pdb(args.pdb, chain_id=args.chain, id=pdb_id)
    except ValueError as e:
        if "Illegal hybrid-36 string" not in str(e):
            raise
        import tempfile
        with open(args.pdb) as fin, tempfile.NamedTemporaryFile("w", delete=False) as fout:
            for line in fin:
                if not line.startswith("ATOM"):
                    continue 
                fout.write(line)

            fixed_pdb = fout.name

        protein_chain = ProteinChain.from_pdb(fixed_pdb, chain_id=args.chain, id=pdb_id)
    finally:
        if fixed_pdb and os.path.exists(fixed_pdb):
            os.remove(fixed_pdb)

    return protein_chain

def main(args):

    config, config_dict = utils.load_config(f'configs/esm3.yml', return_dict=True)
    output_base = handle_name(args)

    model = ESM_model(method='esm3').to(DEVICE)
    model.load_state_dict(torch.load('model/weights/dyna1.pt', map_location=DEVICE), strict=False)
    model.eval()
    seq_input, struct_input = None, None

    # fetch from RCSB
    if args.pdb:
        if len(args.pdb) == 4:
            protein_chain = ProteinChain.from_rcsb(args.pdb.upper(), chain_id=args.chain)
            protein = ESMProtein.from_protein_chain(protein_chain)
        else:
            print(args.pdb)
            if not os.path.isfile(args.pdb):
                exit(f'{args.pdb} does not exist.')
            if not os.path.getsize(args.pdb):
                exit(f'{args.pdb} is empty.')
            pdb_id = args.pdb.split('/')[-1]
            protein_chain = get_pdb_from_upload(args, pdb_id)
            protein = ESMProtein.from_protein_chain(protein_chain)
        encoder = model.model.encode(protein)
        struct_input = encoder.structure[1:-1].unsqueeze(0)
        seq = protein.sequence
        seq_input = encoder.sequence[1:-1].unsqueeze(0)
        sequence_id = seq_input != 4099
        if not args.use_pdb_seq:
            seq_input = None
    if args.sequence:
        if args.pdb and len(seq) != len(args.sequence):
            exit('Length of sequence does not match length of structure input.')
        tokenizer = AutoTokenizer.from_pretrained(f"facebook/esm2_t6_8M_UR50D")
        seq = args.sequence
        token_seq = tokenizer.encode(args.sequence, add_special_tokens=False, return_tensors='np')
        seq_input = torch.from_numpy(token_seq).to(DEVICE)
        sequence_id = seq_input != 4099
    
    sequence_id = seq_input != 4099
    logits = model((seq_input, struct_input), sequence_id)
    p = utils.prob_adjusted(logits).cpu().detach().numpy()

    if args.write_to_pdb:
        out_pdb = os.path.join(args.save_dir, f'{output_base}.pdb')
        protein.to_pdb(out_pdb)
        curr = mda.Universe(out_pdb)
        curr.add_TopologyAttr('bfactors')
        protein_out = curr.select_atoms("protein")
        for residue, p_i in zip(protein_out.residues, p):
            for atom in residue.atoms:
                atom.tempfactor = p_i
        protein_out.write(out_pdb)
    out_df = pd.DataFrame({'position': np.arange(1,len(p)+1), 'residue': np.array(list(seq)), 'p_exchange': p,})
    out_df.to_csv(os.path.join(args.save_dir, f'{output_base}.csv'), index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Example script with integer and string arguments')
    parser.add_argument('--name', type=str, help='name of job')
    parser.add_argument('--pdb', type=str, help='input pdb path or 4-letter code')
    parser.add_argument('--chain', type=str, default='A', help='which chain of the pdb to use, default is chain A')
    parser.add_argument('--sequence', type=str, help='sequence to use, will overide the sequence of the pdb')
    parser.add_argument('--use_pdb_seq', action='store_true', help='whether to use the sequence of the pdb')
    parser.add_argument('--save_dir', type=str, default = '.', help='directory to save outputs')
    parser.add_argument('--write_to_pdb', action='store_true', help='predictions written to the b-factors of the pdb')
    args = parser.parse_args()
    if not (args.sequence or args.pdb):
        exit('Inference requires either a sequence or pdb input')

    if args.sequence:
        alphabets = {'protein': re.compile('^[acdefghiklmnpqrstvwy]*$', re.I)}
        if alphabets['protein'].search(args.sequence) is None:
            exit('Invalid sequence given.')
    main(args)


