
# pip install rdflib
# source venv310/bin/activate

from rdflib import Graph

g = Graph()
g.parse("wbd_populated.owl")

q = """
PREFIX neuro: <http://www.wbd.org/neuro#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?ArticleLabel ?Chemical ?ChemicalLabel ?Disease ?DiseaseLabel
WHERE {
    # Find EDRelationships
    ?EDR a neuro:EDRelationship ;
         neuro:is_evidenced_by ?Article ;
         neuro:has_exposure ?Chemical ;
         neuro:has_disease ?Disease .

    # Filter the ScientificArticle
    ?Article a neuro:ScientificArticle ;
             rdfs:label ?ArticleLabel .

    OPTIONAL { ?Chemical rdfs:label ?ChemicalLabel }
    OPTIONAL { ?Disease  rdfs:label ?DiseaseLabel }

    FILTER (?ArticleLabel = "22231481")
}
LIMIT 20
"""

for row in g.query(q):
    article_label, chemical, chemical_label, disease, disease_label = row

    print("Pubmed ID:", article_label)
    print("Chemical:", chemical_label or chemical)
    print("Disease:", disease_label or disease)
    print("-----")