LOAD CSV WITH HEADERS FROM 'YOUR_DIRECTORY/production_data.csv' AS row
WITH row, date(row.Date) AS date, toInteger(row.Production_Count) AS count
MERGE (f:Factory {name: "Factory 1"})
MERGE (p:Product {id: "ProdA", name: "Widget A"})
MERGE (p)-[:HAS_PRODUCTION_RECORD {date: date, units: count}]->(f);

MERGE (s1:Supplier {name: "Metal Inc."})
MERGE (s2:Supplier {name: "Plastic Co."})
MERGE (d:Distributor {name: "North Region Dist."})

MERGE (f:Factory {name: "Factory 1"})
MERGE (p:Product {id: "ProdA"})

MERGE (s1)-[:SUPPLIES]->(f)
MERGE (s2)-[:SUPPLIES]->(f)
MERGE (f)-[:MANUFACTURES]->(p)
MERGE (p)-[:DISTRIBUTED_BY]->(d);
