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

    class Pollutant(Thing):
        pass

    class ID(Thing):
        pass
    class MeSH(ID):
        pass

    class EDRelationship(Thing):
        pass
    class ScientificArticle(Thing):
        pass

    class TreeCode(ID):   # TreeCode is a subclass of ID
        pass

    class has_id(ObjectProperty):
        domain = [Disease]
        range = [ID]
        pass
        
    class has_exposure(ObjectProperty):
        domain = [EDRelationship]
        range = [Pollutant]
        pass

    class has_disease(ObjectProperty):
        domain = [EDRelationship]
        range = [Disease]
        pass

    class is_evidenced_by(ObjectProperty):
        domain = [EDRelationship]
        range = [ScientificArticle]
        pass


# DiseaseMeSH parsing

df = pd.read_csv("DiseaseMeSH.csv", header=0)

for idx, row in df.iterrows():
    # Clean names for OWL: replace spaces and remove illegal characters
    disease_name = str(row["itemLabel"]).replace(" ", "_")

    # TreeCode values contain dots, so convert them for legal OWL names
    treecode_raw = str(row["treeCode"])
    treecode_name = treecode_raw.replace(".", "_")

    # Create Disease individual
    disease_ind = onto.Disease(disease_name)

    # Create TreeCode individual
    treecode_ind = onto.TreeCode(treecode_name)

    # Link Disease → has_id → TreeCode
    disease_ind.has_id.append(treecode_ind)

# TODO: finish adding more objects and properties

# disease_chems parsing
disease_chems = pd.read_csv("CTD_disease_chems.csv",
                            comment="#",
                            header=None)

for i in range(len(disease_chems)):
    print("Pollutant i : ", disease_chems[1][i], i, "pubmed: ", disease_chems[8][i])
    P1 = onto.Pollutant(disease_chems[1][i])
    D1 = onto.Disease(disease_chems[4][i])
    EDR = onto.EDRelationship(f"{disease_chems[1][i]}_{disease_chems[4][i]}")

    val = disease_chems[8][i]
    if isinstance(val, str):
        for article in val.split("|"):
            SA1 = onto.ScientificArticle(article)
            EDR.is_evidenced_by.append(SA1)

    EDR.has_exposure.append(P1)
    EDR.has_disease.append(D1)
    
# # Save back to OWL
onto.save("wbd.owl")
