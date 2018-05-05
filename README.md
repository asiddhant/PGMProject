# PGMProject

* Usage:
$ python train_srl.py --dataset conll12 --num_epochs 20 
    --pretrnd <PATH_TO_PRETRAINED_EMBEDDINGS> 
    --worddim <EMBEDDING_DIMENSION>
    
Steps:
 
1. Create a folder named wordvectors and put the file "glove.6B.100d.txt" downloaded from 
    http://nlp.stanford.edu/data/glove.6B.zip
2. Untar datasets.tar.gz downloaded from https://drive.google.com/open?id=Anonymized
    and untar it such that you have folder named datasets in parent directory and subfolders conll05, conll12, pbsent etc.
3. Run the above commands and pipe to any output file to store logs.
