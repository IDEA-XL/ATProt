# Towards Stable Representations for Protein Interface Prediction


This is the implementation of the paper **ATProt** submitted to NeurIPS 2024 


## Dependencies

ATProt needs the following environment: 

```shell
python==3.7
numpy==1.22.4
torch-geometric==2.2.0
cuda==10.2
torch==1.11.0
dgl==0.8.1
biopandas==0.4.1
dgllife==0.2.9
joblib==1.1.0
prody==2.4.0
```   

## Dataset Curation

First, generate the required graph structured data for complex with our code. The curator includes two datasets:

- Docking Benchmark 5.5 (DB5.5).
- Database of Interacting Protein Structures (DIPS).

For data preparations, you can choose the configuration as follows:
- **data**. \["dips","db5"\]: Datasets will be processed separately, so please choose one.
- **graph_cutoff**. If the physical distance between two residues in a protein is less than this value, they will be assigned an edge in the KNN graph.
- **graph_max_neighbor**. It means the maximum number of neighbors for each central residue.
- **pocket_cutoff**. If the physical distance between inter-protein residues is less than this value, they will be considered in the pocket.

You can preprocess the raw data as follows for DB5.5:
```
python src.preprocess_raw_data.py -data db5 -graph_cutoff 20 -graph_max_neighbor 10 -pocket_cutoff 8
```
After this, use the following script for generating ESMFold structures
```
python data.esmfold_pro.py 
```


## How to run

You can find a detailed explanation of the parameters in ```./src/utils/args.py```.

To reproduce the results in the paper, you can run the following for **native-bound** and **ESMFold** inference settings.

```
python /src/train.py -inf_data nativebound
```

```
python /src/train.py -inf_data esmfold
```

