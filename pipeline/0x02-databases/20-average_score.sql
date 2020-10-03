-- script that creates a stored procedure
-- compoutes and store average score for a student
delimiter //
CREATE PROCEDURE ComputeAverageScoreForUser(IN user_id INT)
	BEGIN
			UPDATE users
			SET average_score = (SELECT AVG(score)
													 FROM corrections
													 WHERE corrections.user_id = user_id)
			WHERE id = user_id;
	END//
delimiter ;
