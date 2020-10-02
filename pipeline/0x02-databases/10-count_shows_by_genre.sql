-- import the database dump from hbtn_0d_tvshows to your MySQL server
-- write a script that counts all shows contained in database that have at least one genre linked.
-- the database name will be passed as an argument of the mysql command
SELECT tv_genres.name AS genre, COUNT(tv_show_genres.show_id) AS number_of_shows FROM tv_genres
			 JOIN tv_show_genres
			 ON tv_genres.id = tv_show_genres.genre_id
			 GROUP BY genre
			 ORDER BY number_of_shows DESC;