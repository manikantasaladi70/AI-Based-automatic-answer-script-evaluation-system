import sqlite3
conn = sqlite3.connect('smec.db')
try:
    conn.execute('ALTER TABLE users ADD COLUMN name VARCHAR(120)')
    conn.commit()
    print('Column added successfully!')
except Exception as e:
    print('Result:', e)
conn.close()
