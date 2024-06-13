import os
import pickle
from biopandas.pdb import PandasPdb
import pandas as pd
import esm
import torch
from tqdm import tqdm

device = torch.device("cuda:6")
model_esm = esm.pretrained.esmfold_v1()
model_esm = model_esm.eval().to(device)

def seq3to1(residue):

    dit = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
           'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
           'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
           'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
           'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
           'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}
    allowable_set = ['Y', 'R', 'F', 'G', 'I', 'V', 'A', 'W', 'E', 'H', 'C', 'N', 'M', 'D', 'T', 'S', 'K', 'L', 'Q', 'P']
    res_name = residue
    if res_name not in dit.keys():
        res_name = None
    else:
        res_name = dit[res_name]
    return res_name
def filter_residues(residues):
        residues_filtered = []
        for residue in residues:
            df = residue[1]
            Natom = df[df['atom_name'] == 'N']
            alphaCatom = df[df['atom_name'] == 'CA']
            Catom = df[df['atom_name'] == 'C']
            if Natom.shape[0] == 1 and alphaCatom.shape[0] == 1 and Catom.shape[0] == 1:
                residues_filtered.append(residue)
        return residues_filtered

def get_residues_db5(pdb_filename):
    df = PandasPdb().read_pdb(pdb_filename).df['ATOM']
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues = list(df.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    return residues

raw_data_path = './data/benchmark5.5/structures_esmfold/'
reload_mode = 'test'
split_files_path = './data/benchmark5.5/cv/cv_0/'

onlyfiles = [f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path, f))]
code_set = set([file.split('_')[0] for file in onlyfiles])
split_code_set = set()
with open(os.path.join(split_files_path, reload_mode + '.txt'), 'r') as f:
    for line in f.readlines():
        split_code_set.add(line.rstrip())

code_set = code_set & split_code_set
code_list = list(code_set)
str = ''

for code in tqdm(code_list):
    l_res = filter_residues(get_residues_db5(os.path.join(raw_data_path, code + '_l_b.pdb')))
    r_res = filter_residues(get_residues_db5(os.path.join(raw_data_path, code + '_r_b.pdb')))
    l_seq = [ term[1]['resname'].iloc[0] for term in l_res ]
    r_seq = [ term[1]['resname'].iloc[0] for term in r_res ]
    l_s = [seq3to1(s) for s in l_seq]
    r_s = [seq3to1(s) for s in r_seq]
    
    print('processing',code,len(l_s),len(r_s))
    with torch.no_grad():
        output = model_esm.infer_pdb(str.join(l_s))
    with open(os.path.join(raw_data_path, code) + '_l_u.pdb', "w") as f:
        f.write(output)

    with torch.no_grad():
        output = model_esm.infer_pdb(str.join(r_s))
    with open(os.path.join(raw_data_path, code) + '_r_u.pdb', "w") as f:
        f.write(output)
a = 1