import mysql.connector
import json
import os

def insert_mysql_data():
    # MySQL 연결 설정
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="0419",  # 실제 비밀번호
        database="ai_db"
    )
    cursor = conn.cursor()
    
    try:
        # 테이블 이름을 원하는 대로 변경
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS my_temp_table (
            id INT AUTO_INCREMENT PRIMARY KEY,
            project_id INT,
            mapping_data JSON
        )
        ''')
        
        # JSON 파일 읽기
        json_path = "/Users/song-inseop/dev/AI-backend/AI/yolo/framesData.json"
        with open(json_path, "r") as f:
            json_data = json.load(f)
        
        # 데이터 삽입 (테이블 이름 동일하게 변경)
        cursor.execute(
            "INSERT INTO my_temp_table (project_id, mapping_data) VALUES (%s, %s)",
            (1, json.dumps(json_data))
        )
        
        conn.commit()
        print("Sample data inserted successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    insert_mysql_data()