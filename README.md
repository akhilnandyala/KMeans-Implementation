# KMeans-Implementation
Implementation of KMeans Algorithm on unstructured genomic data

You can run the program using the following command:

python kmeans_implementation.py 6 human_skin_microbiome.csv

where
- kmeans_implementation.py is the name of your program, 6 is the size of the tuples, and human_skin_microbiome.csv is
the name of a coma-separated-value file. 
- In that file, the first field is the scientific name of the organism
and the second field is the URL of a compressed FASTA file containing the entire genome of the organism.


You program must download and process all the genomes. Each genome must be transformed into a
frequency vector for all tuples of size l, where l is a command line argument. FASTA is one of the simplest
file format. Such fisle contain one or more sequences. Each sequence begins with a single-line description,
preceded by ’>’ — the information on those lines can be ignored. The data usually spreads over several
lines, where each line is usually (but not always) 80-character long or less. This makes it easier to visualize
the content of the files and was necessary on some older computer systems.
Genomes can be incomplete (i.e. having holes) or made of several chromosomes. Accordingly, each
genome file will have several sequence entries. Youmust read an process all the entries of a given genome
to produce the frequency vector.
Given an input sequence (string) S and a parameter l, Xi is the frequency of the tuple [ACGT]l(i) in S.
Herein, words (tuples) containing symbols other than A, C, G, or T are ignored.
Clearly, two identical or similar sequences will have frequency vectors that are identical or similar. The
further apart the sequences are, as mutations accumulate, the more dissimilar the frequency vectors will
be. This representation has many advantages and limitations. It allows to compare sequences of difierent
lengths. But also, it is tolerant to evolutionary events where the order of segments is rearranged or some
segments are duplicated. However, a significant amount of information is discarded.
Given two input strings T and T, over a four letter alphabet: A, C, G, T.
