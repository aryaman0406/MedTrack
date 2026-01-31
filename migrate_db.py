"""
Database migration script to add missing columns
"""
import sqlite3
import os

def migrate_database():
    # Check for database in both possible locations
    db_paths = ['instance/meds.db', 'meds.db']
    db_path = None
    
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if not db_path:
        print("No database found. Will be created on first run.")
        return
    
    print(f"Migrating database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if expiry_date column exists
        cursor.execute("PRAGMA table_info(reminder)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'expiry_date' not in columns:
            print("Adding expiry_date column to reminder table...")
            cursor.execute("ALTER TABLE reminder ADD COLUMN expiry_date DATE")
            conn.commit()
            print("✓ Successfully added expiry_date column")
        else:
            print("✓ expiry_date column already exists")
        
        print("\nMigration completed successfully!")
        
    except sqlite3.Error as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == '__main__':
    migrate_database()
