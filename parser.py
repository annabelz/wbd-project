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
