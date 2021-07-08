-- Create Table
USE {database};
IF OBJECT_ID('{database}.{schema}.Clustering') IS NOT NULL
	BEGIN
		PRINT 'Database table Clustering exists';
	END
ELSE
    BEGIN
		CREATE TABLE {database}.{schema}.Clustering (
		Feature1 FLOAT(24),
		Feature2 FLOAT(24),
        Target INT
		);
		PRINT 'Database table Clustering created';
	END