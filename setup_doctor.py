import firebase_admin
from firebase_admin import credentials, firestore, auth

try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
except ValueError:
    pass

db = firestore.client()

def create_test_doctor():
    doctor_uid = "test_doctor_uid_123"
    email = "doctor@test.com"
    password = "password123"
    
    try:
        auth.create_user(uid=doctor_uid, email=email, password=password)
        print("Created auth user")
    except:
        print("Auth user already exists")

    db.collection('users').document(doctor_uid).set({
        'name': 'Dr. Test',
        'role': 'doctor',
        'doctorCode': 'DOC123',
        'email': email
    })
    print("Doctor user created/updated in Firestore with code DOC123")

if __name__ == "__main__":
    create_test_doctor()
