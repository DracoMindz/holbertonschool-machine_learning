-- Import the database dump from hbtn_0d_tvshows to your MySQL server
-- Write a script that counts all shows contained in database that have at least one genre linked.              -- The database name will be passed as an argument of the mysql command
SELECT name AS genre, COUNT(*) AS number_of_shows FROM tv_genre GROUP BY genre FROM tv_show_genre JOIN tv_shows  ON genre_id =id GROUP BY genre ORDER BY number_of_shows DESC;
