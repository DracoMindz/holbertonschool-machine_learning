-- script that lists all bands with Glam as their main style
-- ranked by their longevity
-- must set null case to get rid of NULLs
-- must count from current year
SELECT band_name, IF(split is NULL, (2020-formed), (split-formed)) AS lifespan  FROM metal_bands
			 WHERE style like '%Glam Rock%'
			 ORDER BY lifespan DESC;
