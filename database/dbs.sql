 -- Create Database
IF DB_ID('{database}') IS NOT NULL
	BEGIN
		PRINT 'Database {database} exists';
	END
ELSE
	BEGIN
		EXEC('CREATE DATABASE {database}')
		PRINT 'Database {database} created';
	END

GO

-- Create Schema
USE {database};
IF schema_id('{schema}') IS NOT NULL
	BEGIN
		PRINT 'Schema {schema} exists';
	END
ELSE
	BEGIN
		EXEC('CREATE SCHEMA {schema}');
		PRINT 'Schema {schema} created';
	END