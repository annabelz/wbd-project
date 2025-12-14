
# pip install rdflib
# source venv310/bin/activate

from rdflib import Graph, RDF, RDFS, OWL, URIRef, Namespace
import csv

#***********************
# OUTPUT KG STATISTICS #
#***********************

# --------------------------------------------------------
# Load ontology
# --------------------------------------------------------
g = Graph()
g.parse("wbd_populated.owl")

print("\n=== GRAPH STATISTICS ===\n")

# --------------------------------------------------------
# Basic triple + node counts
# --------------------------------------------------------
edges = len(g)

nodes = set()
for s, p, o in g:
    nodes.add(s)
    nodes.add(p)
    nodes.add(o)

print(f"Total edges (triples): {edges}")
print(f"Total nodes: {len(nodes)}")

# --------------------------------------------------------
# Count Classes, Object Properties, Data Properties, Individuals
# --------------------------------------------------------
classes = set(g.subjects(RDF.type, OWL.Class))
object_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
data_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))
individuals = set(g.subjects(RDF.type, OWL.NamedIndividual))

print(f"\nNumber of classes: {len(classes)}")
print(f"Number of object properties: {len(object_props)}")
print(f"Number of data properties: {len(data_props)}")
print(f"Number of individuals: {len(individuals)}")

# --------------------------------------------------------
# Individuals per class
# --------------------------------------------------------
print("\n=== INDIVIDUALS PER CLASS ===\n")

individuals_per_class = {}

for cls in classes:
    inds = set(g.subjects(RDF.type, cls))
    individuals_per_class[cls] = len(inds)

# Sort by descending instance count
for cls, count in sorted(individuals_per_class.items(), key=lambda x: -x[1]):
    if count > 0:
        # Pretty-print local name (after #)
        cls_name = cls.split("#")[-1] if isinstance(cls, URIRef) else str(cls)
        print(f"{cls_name}: {count}")

# --------------------------------------------------------
# Program done
# --------------------------------------------------------
print("\n=== DONE ===\n")

#****************
# QUERY FOR NLP #
#****************

neuro = Namespace("http://www.wbd.org/neuro#")

# --------------------------------------------------------
# SPARQL query: get Article, Disease, Chemical and their labels
# --------------------------------------------------------

q = """
PREFIX neuro: <http://www.wbd.org/neuro#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?ArticleLabel ?DiseaseLabel ?ChemicalLabel
WHERE {
    ?EDR a neuro:EDRelationship .

    ?EDR neuro:is_evidenced_by ?Article .
    ?Article rdfs:label ?ArticleLabel .

    ?EDR neuro:has_disease ?Disease .
    ?Disease rdfs:label ?DiseaseLabel .

    ?EDR neuro:has_exposure ?Chemical .
    ?Chemical rdfs:label ?ChemicalLabel .
}
"""

results = g.query(q)

# --------------------------------------------------------
# Write CSV output
# --------------------------------------------------------
output_file = "query_results.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ArticleLabel", "DiseaseLabel", "ChemicalLabel"])  # header

    for row in results:
        article_label = str(row.ArticleLabel)
        disease_label = str(row.DiseaseLabel)
        chemical_label = str(row.ChemicalLabel)

        writer.writerow([article_label, disease_label, chemical_label])

print(f"\nSaved results to: {output_file}\n")
