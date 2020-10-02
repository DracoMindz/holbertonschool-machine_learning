-- script that lists all records with a score >= 10 in the table
-- listed in decreasing order
SELECT score, name  FROM second_table WHERE score >= 10 ORDER BY score DESC;
