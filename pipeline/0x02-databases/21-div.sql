-- script creates function SafeDiv
-- function divides (and returns) the first by the second
-- or returns 0 if the second number is equal to 0
delimiter //
CREATE FUNCTION SafeDiv(a INT, b INT)
	RETURNS FLOAT
	BEGIN
				RETURN (IF (b = 0, 0, (a/b)));
	END//
delimiter ;