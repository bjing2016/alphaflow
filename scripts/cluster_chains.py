import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--chains', type=str, default='pdb_mmcif.csv')
parser.add_argument('--out', type=str, default='pdb_clusters')
parser.add_argument('--thresh', type=float, default=0.4)
parser.add_argument('--mmseqs_path', type=str, default='mmseqs')
args = parser.parse_args()

import pandas as pd
import os, json, tqdm, random, subprocess, pickle
from collections import defaultdict
from functools import partial
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from collections import defaultdict

def main():
    df = pd.read_csv(args.chains, index_col='name')
    
    sequences = [SeqRecord(Seq(row.seqres), id=name) for name, row in tqdm.tqdm(df.iterrows())]
    SeqIO.write(sequences, ".in.fasta", "fasta")
    cmd = [args.mmseqs_path, 'easy-cluster', '.in.fasta', '.out', '.tmp', '--min-seq-id', str(args.thresh), '--alignment-mode', '1']
    subprocess.run(cmd)#, stdout=open('/dev/null', 'w'))
    f = open('.out_cluster.tsv')
    clusters = []
    for line in f:
        a, b = line.strip().split()
        if a == b:
            clusters.append([])
        clusters[-1].append(b)
    subprocess.run(['rm', '-r', '.in.fasta', '.tmp', '.out_all_seqs.fasta', '.out_rep_seq.fasta', '.out_cluster.tsv'])

    with open(args.out, 'w') as f:
        for clus in clusters:
            f.write(' '.join(clus))
            f.write('\n')
        
    
if __name__ == "__main__":
    main()