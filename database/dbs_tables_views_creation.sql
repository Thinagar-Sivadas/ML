 -- Create Database
IF DB_ID('{create_database}') IS NOT NULL
	BEGIN
		PRINT 'Database {create_database} exists';
	END
ELSE
	BEGIN
		EXEC('CREATE DATABASE {create_database}')
		PRINT 'Database {create_database} created';
	END

GO

-- Create Schema
USE {create_database};
IF schema_id('{create_schema}') IS NOT NULL
	BEGIN
		PRINT 'Schema {create_schema} exists';
	END
ELSE
	BEGIN
		EXEC('CREATE SCHEMA {create_schema}');
		PRINT 'Schema {create_schema} created';
	END

GO

-- Create Table
USE {create_database};
IF OBJECT_ID('{create_database}.{create_schema}.Clustering') IS NOT NULL
	BEGIN
		PRINT 'Database table Clustering exists';
	END
ELSE
    BEGIN
		CREATE TABLE {create_database}.{create_schema}.Clustering (
		Feature1 FLOAT(24),
		Feature2 FLOAT(24),
        Target INT
		);
		PRINT 'Database table Clustering created';
	END

GO

-- Create View
USE {create_database};
IF OBJECT_ID('{create_schema}.query_clustering', 'V') IS NOT NULL
	BEGIN
		PRINT 'View QUERY_Clustering exists'
	END
ELSE
	BEGIN
		EXEC('CREATE VIEW {create_schema}.query_clustering AS
		SELECT * FROM {create_database}.{create_schema}.Clustering')
		PRINT 'View query_clustering created';
	END