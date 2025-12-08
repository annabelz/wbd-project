#-----------------------------------------------------------#
# If running for the first time, set up venv using terminal #
#-----------------------------------------------------------#

# brew install python@3.10
# python3.10 -m venv venv310
# source venv310/bin/activate
# pip install owlready2

##### ACTIVATE VENV BEFORE RUNNING #####

# source venv310/bin/activate

from owlready2 import *
import pandas as pd

onto = get_ontology("wbd.owl").load()

df = pd.read_csv("DiseaseMeSH.csv", header=0)

with onto:
    class Disease(Thing):
        pass

    class ID(Thing):
        pass

    class TreeCode(ID):   # TreeCode is a subclass of ID
        pass

    class has_id(ObjectProperty):
        domain = [Disease]
        range = [ID]
        pass


# --- Create individuals for each row in the dataframe ---

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


# --- Save back to OWL ---
onto.save("wbd.owl")




