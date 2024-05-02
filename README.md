# iGCL Project Source Code
## File Structure
~~
➜  find . -maxdepth 2 -type d
.
./Graph_Classification
./Node_Classification_Citation
./Node_Classification_Citation/citeseer
./Node_Classification_Citation/cora_pubmed
./Node_Classification_Amazon
~~
## Requirements

    torch_geometric
    omegaconf
    warmup_scheduler

## Graph Classification
### IMDB-BINARY
~~
cd ./Graph_Classification; python main_graph.py config/IMDB-BINARY.yml
~~
### IMDB-MULTI
~~
cd ./Graph_Classification; python main_graph.py config/IMDB-MULTI.yml
~~
### IMDB-MUTAG
~~
cd ./Graph_Classification; python main_graph.py config/MUTAG.yml
~~
### NCI1
~~
cd ./Graph_Classification; python main_graph.py config/NCI1.yml
~~
### COLLAB
~~
cd ./Graph_Classification; python main_graph.py config/COLLAB.yml
~~
### PROTEINS
~~
cd ./Graph_Classification; python main_graph.py config/PROTEINS.yml
~~

## Node Classification
### Amazon Photos
~~
cd ./Node_Classification_Amazon
python main.py --dataset Photo --lr 0.005 --epochs 600 --num_samples 3000 --vgae_lr 0.001 --lr_step_rate 0.85
## lower gpu memory cost version
python main.py --dataset Photo --lr 0.005 --epochs 600 --num_samples 500 --vgae_lr 0.001 --lr_step_rate 0.85
~~
### Amazon Computers
~~
cd ./Node_Classification_Amazon
python main.py --dataset Computers --lr 0.001 --epochs 600 --num_samples 3000 --vgae_lr 0.005 --lr_step_rate 0.85
## lower gpu memory cost version
python main.py --dataset Computers --lr 0.001 --epochs 600 --num_samples 500 --vgae_lr 0.005 --lr_step_rate 0.85
~~
### Cora
~~
cd Node_Classification_Citation/cora_pubmed
python reproduce_cora.py
~~
### PubMed
~~
cd Node_Classification_Citation/cora_pubmed
python reproduce_pubmed.py
~~
### Citeseer
~~
cd Node_Classification_Citation/citeseer
python reproduce_citeseer.py
~~
