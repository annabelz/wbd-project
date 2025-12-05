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

onto = get_ontology("wbd.owl").load()

with onto:
    class Disease(Thing):
        pass

    class has_id(ObjectProperty):
        pass


# Create individuals
p1 = onto.Disease("Disease0")

# Save back to OWL
onto.save("wbd.owl")
