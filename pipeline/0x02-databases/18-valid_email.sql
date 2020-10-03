-- script creates a trigger
-- resets attribute Valid_email
-- when the email has been changed
-- use if statement
delimiter //
CREATE TRIGGER ResetEmail BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
	IF OLD.email != NEW.email THEN
		  SET NEW.valid_email = 0;
	END IF;
END;//
delimiter ;
