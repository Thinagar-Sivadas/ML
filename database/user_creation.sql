IF SUSER_ID('{username}') IS NULL
    BEGIN
        CREATE LOGIN {username}
        WITH PASSWORD='{password}',
        CHECK_EXPIRATION=OFF,
        CHECK_POLICY=OFF,
        DEFAULT_DATABASE={database};
    END
ELSE
    BEGIN
        PRINT 'Login exists';
    END

GO

USE {database};
IF USER_ID('{username}') IS NULL
    BEGIN
        CREATE USER {username}
        FOR LOGIN {username}
        WITH DEFAULT_SCHEMA={schema};
    END
ELSE
    BEGIN
        PRINT 'User exists';
    END
ALTER ROLE db_owner ADD MEMBER {username};