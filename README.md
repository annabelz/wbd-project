# wbd-project repo summary

## 1-Parse-Query

### parser.py
Script used to establish knowledge graph - Results/wbd.owl & Results/wbd_populated.csv

### queryer.py
Query populated knowledge graph resulting in output Results/query_results.py

## 2-NLP

### 2.1-ctd-relation-nli.py
Script for applying DeBERTa NLI model to test associative hypotheses for all disease chemical relationships.

### 2.2-Co-occur-RE.py
Script for generating co-occurrence scores and testing relational expressions


## Data

### DiseaseMeSH.csv
Included diseases with associated tree codes and MeSH ID's; generated using Supplementary/SQL-Queries.txt

### CTD_disease_chems.csv
Chemical-gene-disease relationships extracted from CTD's Batch Query with the following columns:
ChemicalName,ChemicalID,CasRN,DiseaseName,DiseaseID,DiseaseCategories,OmimIDs,PubMedIDs,DirectEvidence,InferenceGeneSymbol,InferenceScore

### CTD_disease_chems_curated.csv
Chemical-disease relationships with only 1 row per CDR and no gene symbol inference as in CTD_disease_chems.csv

## Results

### ctd_NLP_results.csv
Output of 2-NLP/2.1-ctd-relation-nli.py

### query_results.csv
Output of 1-Parse-Query/queryer.py

### wbd.owl
Unpopulated XML file containing knowledge graph

### wbd_populated.owl
Fully populated XML file; generated through 1-Parse-Query/parser.py

## Supplementary

### KnowledgeGraph.png

### SQL_Queries.txt

