-- Create View
USE {database};
IF OBJECT_ID('{schema}.query_clustering', 'V') IS NOT NULL
	BEGIN
		PRINT 'View QUERY_Clustering exists'
	END
ELSE
	BEGIN
		EXEC('CREATE VIEW {schema}.query_clustering AS
		SELECT * FROM {database}.{schema}.Clustering')
		PRINT 'View query_clustering created';
	END