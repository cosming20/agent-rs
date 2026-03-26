import asyncio
from agentic_ra.core.config import get_settings
from agentic_ra.stores.postgres.client import PostgresClient
from sqlalchemy import text

async def verify_database():
    settings = get_settings()
    client = PostgresClient(dsn=settings.postgres_dsn_async.get_secret_value())

    print("--- Database Layer Verification ---")
    try:
        async with client.session() as session:
            # Check connection
            result = await session.execute(text("SELECT 1"))
            if result.scalar() == 1:
                print("✅ Connection   | Connected to PostgreSQL")
            
            # Check for migrations/tables (e.g., alembic_version)
            result = await session.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
            tables = [row[0] for row in result.all()]
            print(f"✅ Schema       | Found tables: {', '.join(tables) if tables else 'None'}")
            
    except Exception as e:
        print(f"❌ Verification | Failed: {e}")
        exit(1)
    finally:
        await client.close()

    print("\nDatabase layer verification successful.")

if __name__ == "__main__":
    asyncio.run(verify_database())
