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
    print("Disease name", disease_mesh["itemLabel"][i])
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

    # Link Disease → has_id → TreeCode
    disease_ID_obj.has_treecode.append(disease_tree_obj)
    disease_ID_obj.has_name.append(disease_name_obj)
    disease_ID_obj.has_icd10.append(disease_icd_obj)


# TODO: finish adding more objects and properties

# disease_chems parsing
disease_chems = pd.read_csv("CTD_disease_chems.csv",
                            comment="#",
                            header=None)

# TO CHANGE ONCE LIST HAS BEEN UPDATED: 

# num_entries = len(disease_chems)
num_entries = 20

for i in range(num_entries):
    C1 = onto.Chemical(disease_chems[1][i])
    D1 = onto.Disease(disease_chems[4][i])
    EDR = onto.EDRelationship(f"{disease_chems[1][i]}_{disease_chems[4][i]}")
    ChemID = onto.Name(disease_chems[1][2])

    disease_mesh = disease_chems[5][i].replace('MESH:', '')
    DiseaseID = onto.Name(disease_mesh)

    val = disease_chems[8][i]
    if isinstance(val, str):
        for article in val.split("|"):
            SA1 = onto.ScientificArticle(article)
            EDR.is_evidenced_by.append(SA1)

    EDR.has_exposure.append(C1)
    EDR.has_disease.append(D1)
    C1.has_name.append(ChemID)
    
# # Save back to OWL
onto.save("wbd.owl")
