from esm.sdk.api import ESMProtein
from esm.models.esm3 import ESM3
import os
import torch
import pickle
import numpy as np
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

esm3 = ESM3.from_pretrained("esm3_sm_open_v1").to(DEVICE,non_blocking=True).to(torch.float32)
esm3.eval()

for name in os.listdir('data/mBMRB_data'):
    if '_' in name and 'CPMG' not in name:
        n = name.split('_')[0]
        n2 = name.split('_')[-1].replace('.pkl', '')
        pdb_path = f'/scratch/users/gelnesr/esm_pdbs/{n}_{n2}.pdb'
    else:
        n = name.split('.')[0]
        pdb_path = f'/scratch/users/gelnesr/esm_pdbs/{n}.pdb'
    
    pkl_fname = f"/scratch/users/gelnesr/esm3_data/{name}"

    if os.path.exists(pkl_fname):
        continue
    
    protein = ESMProtein.from_pdb(pdb_path)
    encoder = esm3.encode(protein)
    seq = encoder.sequence.cpu().detach()[1:-1][:700]
    struct = encoder.structure.cpu().detach()[1:-1][:700]
    
    sequence_tokens = np.full(700, 1, dtype=np.int32) ## sequence pad token is 1
    structure_tokens = np.full(700, 4099, dtype=np.int32) ## structure pad token is 4099
            
    sequence_tokens[:len(seq)] = seq
    structure_tokens[:len(struct)] = struct

    sequence_id = sequence_tokens != 1

    obj ={'name': name, 'len': len(seq), 'seq_tokens': sequence_tokens, 
            'struct_tokens': structure_tokens, 'sequence_id': sequence_id}
    
    with open(pkl_fname, 'wb') as f:
        pickle.dump(obj, f)

    print(pdb_path)