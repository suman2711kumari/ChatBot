import sqlite3
from datetime import datetime, timezone, timedelta

# Connect to database
conn = sqlite3.connect('chatbot.db')
c = conn.cursor()

# Add some test messages with different timestamps
test_data = [
    ('user_1', 'I feel sad today', 'I\'m sorry to hear you\'re feeling sad. Would you like to talk more about it?', 3.5),
    ('user_2', 'I am anxious about work', 'Anxiety can be overwhelming. Try to take deep breaths and focus on the present moment.', 2.8),
    ('user_1', 'Everything seems hopeless', 'It sounds like you\'re going through a tough time. Remember, you\'re not alone.', 6.2),
    ('user_3', 'I feel lonely', 'Feeling lonely is hard. I\'m here to listen whenever you want to share.', 4.1),
]

# Insert test data with timestamps from the last few hours
for i, (user_id, message, reply, score) in enumerate(test_data):
    timestamp = datetime.now(timezone.utc) - timedelta(hours=i+1)
    c.execute('''
        INSERT INTO messages (user_id, message, reply, depression_score, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, message, reply, score, timestamp.isoformat()))

conn.commit()
conn.close()
print("Test data added successfully!")