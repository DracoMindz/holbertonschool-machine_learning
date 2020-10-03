-- script creates a stored procedure AddBonus
-- AddBonus adds a new correction for a student
-- AddBonus takes three inputs
-- DUAL is a "dummy" table
delimiter //
CREATE PROCEDURE AddBonus(IN user_id INT,
													IN project_name VARCHAR(255),
													IN score INT)
	BEGIN
		INSERT INTO project(name)
		SELECT project_name FROM DUAL
		WHERE NOT EXISTS (SELECT * FROM projects
		 									WHERE name = project_name
											LIMIT 1);

		INSERT INTO corrections(user_id, project_id, score)
		VALUES(user_id,
					 (SELECT id FROM projects WHERE name = project_name),
					  score);
	END//
delimiter ;