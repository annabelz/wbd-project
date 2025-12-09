
# pip install rdflib
# source venv310/bin/activate

from rdflib import Graph

g = Graph()
g.parse("wbd_populated.owl")

q = """
PREFIX neuro: <http://www.wbd.org/neuro#>

SELECT ?EDRelationship ?DiseaseLabel ?NameLabel ?ChemicalLabel
WHERE {
    ?EDRelationship a neuro:EDRelationship .
    ?EDRelationship neuro:has_exposure ?Chemical .
    ?EDRelationship neuro:has_disease ?Disease .
    ?Disease neuro:has_name ?Name .
    
    OPTIONAL { ?Disease rdfs:label ?DiseaseLabel }
    OPTIONAL { ?Name rdfs:label ?NameLabel }
    OPTIONAL { ?Chemical rdfs:label ?ChemicalLabel }
}
LIMIT 20
"""

for edr, disease_label, name_label, chemical_label in g.query(q):
    print(f"Disease label: {disease_label}")
    print(f"Disease name label: {name_label}")
    print(f"Chemical label: {chemical_label}")
    print(f"EDRelationship: {edr}")
    print("-----")