#-----------------------------------------------------------#
# If running for the first time, set up venv using terminal #
#-----------------------------------------------------------#

# brew install python@3.10
# python3.10 -m venv venv310
# source venv310/bin/activate
# pip install owlready2
# pip install pandas

##### ACTIVATE VENV BEFORE RUNNING #####

# source venv310/bin/activate

import pandas as pd
from owlready2 import *

onto = get_ontology("wbd.owl").load()

with onto:
    class Disease(Thing):
        pass

    class Chemical(Thing):
        pass

    class Gene(Thing):
        pass
    class Name(Thing):
        pass
    class EDRelationship(Thing):
        pass
    class ScientificArticle(Thing):
        pass

    class TreeCode(Thing):  
        pass

    class Icd10(Thing):  
        pass

    class has_treecode(ObjectProperty):
        domain = [Disease]
        range = [TreeCode]
        pass

    class has_icd10(ObjectProperty):
        domain = [Disease]
        range = [Icd10]
        pass

    class has_name(ObjectProperty):
        domain = [Chemical, Disease]
        range = [Name]
        pass
        
    class has_exposure(ObjectProperty):
        domain = [EDRelationship]
        range = [Chemical]
        pass

    class has_disease(ObjectProperty):
        domain = [EDRelationship]
        range = [Disease]
        pass

    class is_evidenced_by(ObjectProperty):
        domain = [EDRelationship]
        range = [ScientificArticle]
        pass

    class is_associated_with(ObjectProperty):
        domain = [EDRelationship]
        range = [Gene]
        pass

    # TODO: add is_child_of and is_parent_of if relevant

# DiseaseMeSH parsing

disease_mesh = pd.read_csv("DiseaseMeSH.csv", header=0)

for i in range(len(disease_mesh)):
    # extract vars
    disease_name = disease_mesh["itemLabel"][i].replace(" ", "_")
    disease_ID = disease_mesh["meshID"][i]
    disease_tree = disease_mesh["treeCode"][i].replace(".", "_")
    disease_icd = disease_mesh["icd10"][i]

    # create objects per row
    disease_ID_obj = onto.Disease(disease_ID)
    disease_name_obj = onto.Name(disease_name)
    disease_tree_obj = onto.TreeCode(disease_tree)
    disease_icd_obj = onto.Icd10(disease_icd)
    
    # create relationships
    disease_ID_obj.has_treecode.append(disease_tree_obj)
    disease_ID_obj.has_name.append(disease_name_obj)
    disease_ID_obj.has_icd10.append(disease_icd_obj)

# disease_chems parsing
disease_chems = pd.read_csv("CTD_disease_chems.csv",
                            comment="#",
                            header=None)

# TO CHANGE ONCE LIST HAS BEEN UPDATED: 

num_entries = len(disease_chems)
for i in range(num_entries):
    # extract vars
    chem_name = onto.Name(disease_chems[1][i].replace(" ", "_"))
    chem_id = onto.Chemical(disease_chems[2][i])
    disease_name = onto.Name(disease_chems[4][i].replace(" ", "_"))
    disease_id = onto.Disease(disease_chems[5][i].replace('MESH:', ''))
    edr_name = onto.EDRelationship(f"{chem_id}_{disease_id}")
    gene = onto.Gene(disease_chems[10][i])

    pubmed_id_list = disease_chems[8][i]
    if isinstance(pubmed_id_list, str):
        for article in pubmed_id_list.split("|"):
            SA1 = onto.ScientificArticle(article)
            edr_name.is_evidenced_by.append(SA1)

    edr_name.has_exposure.append(chem_id)
    edr_name.has_disease.append(disease_id)
    edr_name.is_associated_with.append(gene)
    chem_id.has_name.append(chem_name)
    disease_id.has_name.append(disease_name)
    
# # Save back to OWL
onto.save("wbd_populated.owl")
