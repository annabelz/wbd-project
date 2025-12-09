
# pip install rdflib
# source venv310/bin/activate

from rdflib import Graph

g = Graph()
g.parse("wbd_populated.owl")

q = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?cls ?label WHERE {
    ?cls a <http://www.w3.org/2002/07/owl#Class> .
    OPTIONAL { ?cls rdfs:label ?label }
}
"""

for row in g.query(q):
    print(row)
