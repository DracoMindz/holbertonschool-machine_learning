-- script that lists all shows contained in hbtn_0d_tvshows without a genre linked.
-- The database name will be passed as an argument of the mysql command
SELECT tv_shows.title, tv_show_genre_id FROM tv_shows LEFT JOIN tv_show_genres ON id = show_id WHERE genre.id = NULL ORDER BY title ASC, genre_id ASC;