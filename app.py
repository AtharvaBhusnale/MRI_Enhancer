import os
import io
import time
import threading
import torch
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import pydicom

from flask import Flask, request, jsonify, send_file, render_template, session, send_from_directory
from flask_cors import CORS 
from fpdf import FPDF
from datetime import datetime 
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import firebase_admin
from firebase_admin import credentials, auth, firestore

from bot import Chatbot 
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from segmentation_service import SegmentationService
from cryptography.fernet import Fernet

#---------------------------------------------------
# 1. INITIALIZE ALL SERVICES
#---------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key_12345'
CORS(app) 

UPLOAD_FOLDER = 'uploads'
ORIGINAL_FOLDER = os.path.join(UPLOAD_FOLDER, 'original')
ENHANCED_FOLDER = os.path.join(UPLOAD_FOLDER, 'enhanced')
SEGMENTED_FOLDER = os.path.join(UPLOAD_FOLDER, 'segmented')
os.makedirs(ORIGINAL_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- ENCRYPTION SETUP ---
KEY_FILE = 'secret.key'
if not os.path.exists(KEY_FILE):
    key = Fernet.generate_key()
    with open(KEY_FILE, 'wb') as key_file:
        key_file.write(key)
else:
    with open(KEY_FILE, 'rb') as key_file:
        key = key_file.read()

cipher_suite = Fernet(key)

def encrypt_file(file_path):
    """Encrypts a file in place."""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        encrypted_data = cipher_suite.encrypt(data)
        with open(file_path, 'wb') as f:
            f.write(encrypted_data)
        print(f"🔒 Encrypted: {file_path}")
    except Exception as e:
        print(f"Error encrypting {file_path}: {e}")

def decrypt_file(file_path):
    """Returns decrypted bytes of a file."""
    try:
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
        return cipher_suite.decrypt(encrypted_data)
    except Exception as e:
        print(f"Error decrypting {file_path}: {e}")
        return None

try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred) 
    db = firestore.client()
    print("✅ Firebase Admin SDK loaded.")
except FileNotFoundError:
    print("🔴 Error: 'serviceAccountKey.json' not found.")
    db = None
except ValueError:
    print("Firebase Admin SDK already initialized.")
    db = firestore.client()

def log_audit_event(user_uid, action, resource_id=None, details=None):
    """Logs an event to the audit_logs collection."""
    if not db: return
    try:
        audit_entry = {
            'timestamp': firestore.SERVER_TIMESTAMP,
            'userUid': user_uid,
            'action': action,
            'resourceId': resource_id,
            'details': details
        }
        db.collection('audit_logs').add(audit_entry)
        print(f"📝 Audit Log: {action} by {user_uid}")
    except Exception as e:
        print(f"Error logging audit event: {e}")

#---------------------------------------------------
# 2. LOAD AI MODELS
#---------------------------------------------------
# --- CONFIGURATION ---
GEMINI_API_KEY = "AIzaSyCuaW2lBS0JvN44J3Njzl2kYEhMiyRCw_o"  # Replace with your actual key
# ---------------------

chatbot = Chatbot(api_key=GEMINI_API_KEY, knowledge_base_path="knowledge_base.json")
print("✅ Chatbot (LLM) loaded.")

def load_enhancer_model():
    model_path = os.path.join('weights', 'RealESRGAN_x4plus.pth')
    if not os.path.exists(model_path):
        print("🔴 Error: Model file not found")
        return None
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Loading Enhancer model onto device: {device}...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4, model_path=model_path, model=model,
        dni_weight=None, tile=0, tile_pad=10, pre_pad=0, half=False, device=device
    )
    print("✅ Enhancer model loaded.")
    return upsampler

upsampler = load_enhancer_model()

# Initialize Segmentation Service
segmentation_service = SegmentationService(model_path='weights/unet_brain.pth')

# Helper function for DICOM processing
def process_dicom(file_stream):
    try:
        ds = pydicom.dcmread(file_stream)
        pixel_array = ds.pixel_array.astype(float)
        
        # Normalize to 0-255
        scaled_image = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        scaled_image = np.uint8(scaled_image)
        
        return Image.fromarray(scaled_image).convert('RGB')
    except Exception as e:
        print(f"Error processing DICOM: {e}")
        return None

#---------------------------------------------------
# 2.1 EMAIL NOTIFICATION HELPER
#---------------------------------------------------
def send_email_notification(doctor_email, patient_name, patient_id):
    # --- CONFIGURATION ---
    # REPLACE THESE WITH YOUR ACTUAL GMAIL CREDENTIALS
    sender_email = "a.bhusnale193@gmail.com" 
    sender_password = "evpd gcwr rdjp kmye" 
    # ---------------------
    
    if "YOUR_GMAIL_ADDRESS" in sender_email:
        print("⚠️ Email notification skipped: Credentials not set in app.py.")
        return

    subject = f"New MRI Upload: {patient_name}"
    
    # Professional HTML Template
    body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden;">
          <div style="background-color: #008080; padding: 20px; text-align: center;">
            <h2 style="color: white; margin: 0;">Medical Enhancement Notification</h2>
          </div>
          <div style="padding: 30px;">
            <p>Dear Doctor,</p>
            <p>A new MRI scan has been uploaded and processed for your review.</p>
            
            <div style="background-color: #f9f9f9; padding: 15px; border-left: 4px solid #008080; margin: 20px 0;">
              <p style="margin: 5px 0;"><strong>Patient Name:</strong> {patient_name}</p>
              <p style="margin: 5px 0;"><strong>Patient ID:</strong> {patient_id}</p>
              <p style="margin: 5px 0;"><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            
            <p>Please log in to your dashboard to view the enhanced image and download the detailed report.</p>
            
            <div style="text-align: center; margin-top: 30px;">
              <a href="http://127.0.0.1:5000/" style="background-color: #008080; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; font-weight: bold;">Go to Dashboard</a>
            </div>
          </div>
          <div style="background-color: #f4f4f4; padding: 15px; text-align: center; font-size: 12px; color: #666;">
            <p>&copy; 2025 AI Medical Assistant. All rights reserved.</p>
            <p>This is an automated message. Please do not reply.</p>
          </div>
        </div>
      </body>
    </html>
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = doctor_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, doctor_email, text)
        server.quit()
        print(f"📧 Notification sent to {doctor_email}")
    except Exception as e:
        print(f"🔴 Failed to send email: {e}")

#---------------------------------------------------
# 3. AUTHENTICATION & ROLE CHECKING
#---------------------------------------------------
def get_user_data(uid):
    if not db: return 'patient', 'Patient' 
    try:
        user_doc_ref = db.collection('users').document(uid)
        user_doc = user_doc_ref.get()
        if user_doc.exists:
            data = user_doc.to_dict()
            return data.get('role', 'patient'), data.get('name', 'Patient ' + uid[:5])
        else:
            user_doc_ref.set({'role': 'patient', 'name': 'New Patient'})
            return 'patient', 'New Patient'
    except Exception as e:
        print(f"Error getting user data: {e}")
        return 'patient', 'Patient'

def check_auth(role_required=None):
    def decorator(f):
        def wrapper(*args, **kwargs):
            id_token = request.headers.get('Authorization')
            if not id_token:
                return jsonify({'error': 'No auth token provided'}), 401
            try:
                id_token = id_token.split('Bearer ')[1]
                decoded_token = auth.verify_id_token(id_token)
                uid = decoded_token['uid']
                user_role, user_name = get_user_data(uid)
                session['uid'] = uid
                session['role'] = user_role
                session['name'] = user_name
                if role_required and user_role not in role_required:
                    return jsonify({'error': 'Insufficient permissions'}), 403
                return f(*args, **kwargs)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        wrapper.__name__ = f.__name__ + '_wrapper'
        return wrapper
    return decorator

#---------------------------------------------------
# 4. API ENDPOINTS
#---------------------------------------------------

@app.route('/api/set_profile', methods=['POST'])
@check_auth() 
def set_profile():
    data = request.json
    name = data.get('name')
    if not name: return jsonify({'error': 'Name is required'}), 400
    try:
        db.collection('users').document(session['uid']).update({'name': name})
        session['name'] = name # Update session immediately
        return jsonify({'success': True, 'name': name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_profile', methods=['GET'])
@check_auth()
def get_profile():
    """Returns the current user's profile from server session/db."""
    try:
        # User data is already loaded into session by the @check_auth decorator
        return jsonify({
            'uid': session.get('uid'),
            'role': session.get('role', 'patient'),
            'name': session.get('name', 'Unknown'),
            # We might want doctorCode if available, checking DB again or storing in session
        })
    except Exception as e:
         return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
@check_auth(role_required=['patient']) 
def handle_chat():
    data = request.json
    user_message = data.get('message', '').lower()

    if user_message == 'init': # Handle the init call from frontend
        session['severity_score'] = 0
        session['symptoms_discussed'] = []
        return jsonify({'reply': 'Session initialized.'})

    if 'severity_score' not in session: session['severity_score'] = 0
    if 'symptoms_discussed' not in session: session['symptoms_discussed'] = []
    if user_message == 'stop':
        bot_reply = chatbot.generate_summary(session['severity_score'], session['symptoms_discussed'])
        session['chat_summary'] = bot_reply # Store summary in session
        session['severity_score'] = 0
        session['symptoms_discussed'] = []
    else:
        bot_reply, new_score, new_symptom = chatbot.process_message(user_message)
        if new_symptom and new_symptom not in session['symptoms_discussed']:
            session['symptoms_discussed'].append(new_symptom)
        session['severity_score'] += new_score
    return jsonify({'reply': bot_reply})

@app.route('/api/upload_image', methods=['POST'])
@check_auth(role_required=['patient']) 
def upload_image():
    if upsampler is None: return jsonify({'error': 'Model not loaded'}), 500
    if 'image' not in request.files: return jsonify({'error': 'No image file provided'}), 400
    
    # --- Get Doctor Code from the form ---
    doctor_code = request.form.get('doctorCode')
    if not doctor_code:
        return jsonify({'error': "Doctor's Code is required."}), 400
    # ---

    file = request.files['image']
    uid = session['uid']
    # Use name from form if provided (fixes race condition), else fallback to session
    patient_name = request.form.get('patientName', session['name'])
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        # --- Find the doctor's UID from their code ---
        doctor_uid = None
        # Query Firestore to find the user with this doctorCode
        docs = db.collection('users').where('doctorCode', '==', doctor_code).limit(1).stream()
        for doc in docs:
            doctor_uid = doc.id # Get the doctor's main user ID
        
        if not doctor_uid:
            return jsonify({'error': 'Invalid Doctor Code. No doctor found.'}), 404
        # ---
        
        base_filename = secure_filename(file.filename)
        unique_timestamp = int(time.time())
        original_filename = f"{uid}_{unique_timestamp}_{base_filename}"
        
        # Handle DICOM or Standard Image
        if base_filename.lower().endswith('.dcm'):
            original_img = process_dicom(file.stream)
            if original_img is None:
                return jsonify({'error': 'Invalid or unreadable DICOM file'}), 400
            # Save as PNG for consistency
            original_filename = original_filename.replace('.dcm', '.png')
        else:
            original_img = Image.open(file.stream).convert('RGB')
            
        original_img_np = np.array(original_img)
        original_path = os.path.join(ORIGINAL_FOLDER, original_filename)
        original_img.save(original_path)
        encrypt_file(original_path) # <--- ENCRYPT ORIGINAL
        
        original_url = f"/api/serve_image/uploads/original/{original_filename}" # Point to secure endpoint
        
        output_np, _ = upsampler.enhance(original_img_np, outscale=4)
        output_img = Image.fromarray(output_np)
        
        enhanced_filename = f"{uid}_{unique_timestamp}_enhanced.png"
        enhanced_path = os.path.join(ENHANCED_FOLDER, enhanced_filename)
        output_img.save(enhanced_path)
        encrypt_file(enhanced_path) # <--- ENCRYPT ENHANCED
        
        enhanced_url = f"/api/serve_image/uploads/enhanced/{enhanced_filename}" # Point to secure endpoint

        # --- Save the log with the assigned doctor's UID ---
        log_entry = {
            'patientUid': uid,
            'patientName': patient_name,
            'assignedDoctorUid': doctor_uid, 
            'originalImageUrl': original_url,
            'enhancedImageUrl': enhanced_url,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'originalFilename': base_filename,
            'symptomSummary': session.get('chat_summary', 'No summary available.') # Save summary to log
        }
        db.collection('mri_uploads').add(log_entry)
        
        print(f"✅ Created log for patient {uid} assigned to doctor {doctor_uid}.")
        
        # --- Send Email Notification ---
        try:
            # Fetch doctor's email from Auth
            doctor_user = auth.get_user(doctor_uid)
            if doctor_user.email:
                # Run email in background thread to avoid blocking response
                email_thread = threading.Thread(target=send_email_notification, args=(doctor_user.email, patient_name, uid))
                email_thread.start()
                print(f"📧 Email task started in background for {doctor_user.email}")
        except Exception as e:
            print(f"⚠️ Could not send email notification: {e}")
        # ---
        
        return jsonify({
            'success': True, 
            'message': 'Image uploaded and enhancement complete.',
            'original_url': original_url,
            'enhanced_url': enhanced_url
        })
    except Exception as e:
        print(f"🔴 Error during upload/enhancement: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/segment_image', methods=['POST'])
@check_auth(role_required=['doctor'])
def segment_image():
    data = request.json
    log_id = data.get('log_id')
    
    if not log_id:
        return jsonify({'error': 'Log ID is required'}), 400
        
    try:
        # Fetch log to get image path
        doc_ref = db.collection('mri_uploads').document(log_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({'error': 'Log entry not found'}), 404
            
        log_data = doc.to_dict()
        image_url = log_data.get('enhancedImageUrl') # Segment the enhanced image
        
        if not image_url:
             return jsonify({'error': 'Enhanced image URL missing in log'}), 400

        # Convert URL to local path
        # Remove /api/serve_image/ prefix if present
        if 'serve_image' in image_url:
            image_url = image_url.replace('/api/serve_image/', '')
            
        if image_url.startswith('/'): image_url = image_url[1:]
        local_path = os.path.abspath(image_url)
        
        if not os.path.exists(local_path):
            return jsonify({'error': 'Image file not found'}), 404
            
        # Decrypt for processing
        decrypted_bytes = decrypt_file(local_path)
        if not decrypted_bytes:
             return jsonify({'error': 'Decryption failed'}), 500
             
        # Save temp decrypted file for segmentation service (it expects a path)
        temp_path = local_path + ".temp.png"
        with open(temp_path, 'wb') as f:
            f.write(decrypted_bytes)
            
        filename = os.path.basename(local_path)
        segmented_filename = f"seg_{filename}"
        segmented_path = os.path.join(SEGMENTED_FOLDER, segmented_filename)
        
        success = segmentation_service.segment(temp_path, segmented_path)
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if success:
            encrypt_file(segmented_path) # <--- ENCRYPT SEGMENTED
            segmented_url = f"/api/serve_image/uploads/segmented/{segmented_filename}"
            
            # Update Firestore
            doc_ref.update({'segmentedImageUrl': segmented_url})
            
            log_audit_event(session['uid'], "SEGMENT_IMAGE", log_id)
            
            return jsonify({
                'success': True, 
                'segmented_url': segmented_url
            })
        else:
            return jsonify({'error': 'Segmentation failed'}), 500
            
    except Exception as e:
        print(f"Error in segment_image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient_log', methods=['GET'])
@check_auth(role_required=['doctor']) 
def get_patient_log():
    try:
        log_entries = []
        
        # --- Filter logs by the logged-in doctor's UID ---
        doctor_uid = session['uid']
        # Note: Removed order_by from Firestore query to avoid "requires an index" error.
        # We will sort in Python instead.
        docs = db.collection('mri_uploads').where('assignedDoctorUid', '==', doctor_uid).stream()
        # ---
        
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            # Convert timestamp to datetime object for sorting if it's not already
            log_entries.append(data)
            
        # Sort in Python (descending order by timestamp)
        log_entries.sort(key=lambda x: x.get('timestamp', 0), reverse=True)

        # Format timestamp for display
        for entry in log_entries:
            if 'timestamp' in entry and entry['timestamp']:
                 # Check if it's a datetime object or string
                 ts = entry['timestamp']
                 if hasattr(ts, 'strftime'):
                    entry['timestamp'] = ts.strftime("%Y-%m-%d %H:%M")
        
        log_audit_event(session['uid'], "VIEW_LOGS")
                 
        return jsonify({'log': log_entries})
    except Exception as e:
        print(f"🔴 Error in get_patient_log: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_log/<log_id>', methods=['DELETE'])
@check_auth(role_required=['doctor'])
def delete_log(log_id):
    try:
        # 1. Fetch the log entry
        doc_ref = db.collection('mri_uploads').document(log_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({'error': 'Log entry not found'}), 404
        
        data = doc.to_dict()
        
        # 2. Delete files
        def delete_file(url_path):
            if not url_path: return
            # Handle secure URLs
            if 'serve_image' in url_path:
                url_path = url_path.replace('/api/serve_image/', '')
                
            if url_path.startswith('/'): url_path = url_path[1:]
            local_path = os.path.abspath(url_path)
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                    print(f"Deleted file: {local_path}")
                except Exception as e:
                    print(f"Error deleting file {local_path}: {e}")

        delete_file(data.get('originalImageUrl'))
        delete_file(data.get('enhancedImageUrl'))
        delete_file(data.get('segmentedImageUrl'))
        
        # 3. Delete Firestore document
        doc_ref.delete()
        
        log_audit_event(session['uid'], "DELETE_LOG", log_id)
        
        return jsonify({'success': True, 'message': 'Log deleted successfully'})

    except Exception as e:
        print(f"Error deleting log: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_report/<log_id>', methods=['GET'])
@check_auth(role_required=['doctor'])
def download_report(log_id):
    try:
        # 1. Fetch the log entry
        doc_ref = db.collection('mri_uploads').document(log_id)
        doc = doc_ref.get()
        if not doc.exists:
            return "Log entry not found", 404
        
        data = doc.to_dict()
        
        # Fetch Doctor's Name
        doctor_name = "Unknown"
        if 'assignedDoctorUid' in data:
            doctor_doc = db.collection('users').document(data['assignedDoctorUid']).get()
            if doctor_doc.exists:
                doctor_name = doctor_doc.to_dict().get('name', 'Unknown')

        # 2. Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # --- Header ---
        pdf.set_font("Arial", 'B', 20)
        pdf.cell(0, 15, "Hospital Name / Project Name", ln=True, align='C')
        pdf.ln(5)
        
        # Draw a line
        pdf.line(10, 30, 200, 30)
        pdf.ln(10)
        
        # --- Patient & Report Info ---
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(40, 8, "Patient Name:", 0, 0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 8, data.get('patientName', 'Unknown'), ln=True)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(40, 8, "Patient ID:", 0, 0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 8, data.get('patientUid', 'Unknown'), ln=True)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(40, 8, "Doctor:", 0, 0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 8, f"Dr. {doctor_name}", ln=True)

        timestamp = data.get('timestamp')
        if timestamp:
            dt_object = timestamp
            if hasattr(timestamp, 'date'): 
                 dt_object = timestamp
            date_str = dt_object.strftime("%Y-%m-%d %H:%M:%S")
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(40, 8, "Date:", 0, 0)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 8, date_str, ln=True)
            
        pdf.ln(5)
        
        # --- Symptom Summary ---
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Symptom Summary", ln=True, align='L')
        pdf.set_font("Arial", '', 11)
        summary_text = data.get('symptomSummary', 'No summary available.')
        # Clean up markdown if present (basic)
        summary_text = summary_text.replace('**', '').replace('*', '-')
        pdf.multi_cell(0, 6, summary_text)
        pdf.ln(10)

        # --- Images Section ---
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "MRI Scan Comparison", ln=True, align='L')
        pdf.ln(5)
        
        def get_local_path(url_path):
            if not url_path: return None
            if url_path.startswith('/'): url_path = url_path[1:]
            return os.path.abspath(url_path)

        original_path = get_local_path(data.get('originalImageUrl').replace('/api/serve_image/', ''))
        enhanced_path = get_local_path(data.get('enhancedImageUrl').replace('/api/serve_image/', ''))
        
        # Helper to get decrypted bytes for PDF
        def get_image_for_pdf(path):
            if not path or not os.path.exists(path): return None
            decrypted = decrypt_file(path)
            
            # Fallback for unencrypted files
            if not decrypted:
                try:
                    with open(path, 'rb') as f:
                        decrypted = f.read()
                except Exception as e:
                    print(f"Error reading file for PDF fallback: {e}")
                    return None

            if not decrypted: return None

            # Validate Image Header (PNG or JPG)
            is_valid = False
            if len(decrypted) > 8:
                header = decrypted[:8]
                if header.startswith(b'\x89PNG\r\n\x1a\n'):
                    is_valid = True
                elif header.startswith(b'\xff\xd8\xff'):
                    is_valid = True
            
            if not is_valid:
                print(f"⚠️ Invalid image header for {path}. Skipping PDF inclusion.")
                return None

            # Save to temp file for FPDF
            temp_name = path + ".pdf_temp.png"
            with open(temp_name, 'wb') as f:
                f.write(decrypted)
            return temp_name

        temp_orig = get_image_for_pdf(original_path)
        temp_enh = get_image_for_pdf(enhanced_path)
        
        # Side-by-Side Layout
        y_start = pdf.get_y()
        
        # Ensure we don't go off page
        if y_start > 200:
            pdf.add_page()
            y_start = 20
        
        if temp_orig:
            pdf.image(temp_orig, x=10, y=y_start, w=90)
            pdf.set_xy(10, y_start + 95)
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(90, 10, "Original Scan", align='C')
            
        if temp_enh:
            pdf.image(temp_enh, x=110, y=y_start, w=90)
            pdf.set_xy(110, y_start + 95)
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(90, 10, "AI Enhanced Scan", align='C')
            
        pdf.ln(20)
        
        # Clean up temp files
        if temp_orig and os.path.exists(temp_orig): os.remove(temp_orig)
        if temp_enh and os.path.exists(temp_enh): os.remove(temp_enh)
        
        # Output
        buffer = io.BytesIO()
        pdf_output = pdf.output(dest='S').encode('latin-1')
        buffer.write(pdf_output)
        buffer.seek(0)
        
        log_audit_event(session['uid'], "DOWNLOAD_REPORT", log_id)
        
        return send_file(buffer, as_attachment=True, download_name=f"Report_{data.get('patientName')}.pdf", mimetype='application/pdf')

    except Exception as e:
        print(f"Error generating report: {e}")
        return f"Error generating report: {e}", 500

# --- NEW: Secure Image Serving ---
@app.route('/api/serve_image/<path:filename>', methods=['GET'])
@check_auth() # Requires login to view images
def serve_image(filename):
    try:
        # Prevent directory traversal
        if '..' in filename or filename.startswith('/'):
            return "Invalid path", 400
            
        # Construct absolute path
        file_path = os.path.abspath(filename)
        
        if not os.path.exists(file_path):
            return "File not found", 404
            
        # Decrypt
        decrypted_bytes = decrypt_file(file_path)
        
        # Fallback for unencrypted files
        if not decrypted_bytes:
            try:
                with open(file_path, 'rb') as f:
                    decrypted_bytes = f.read()
            except Exception as e:
                return f"Error reading file: {e}", 500
            
        # Log Access
        log_audit_event(session['uid'], "VIEW_IMAGE", filename)
        
        return send_file(
            io.BytesIO(decrypted_bytes),
            mimetype='image/png' # Assuming PNG for now
        )
    except Exception as e:
        print(f"Error serving image: {e}")
        return str(e), 500

# --- NEW: Delete My Data (Patient Rights) ---
@app.route('/api/delete_my_data', methods=['DELETE'])
@check_auth(role_required=['patient'])
def delete_my_data():
    uid = session['uid']
    try:
        # 1. Find all logs for this patient
        docs = db.collection('mri_uploads').where('patientUid', '==', uid).stream()
        
        count = 0
        for doc in docs:
            data = doc.to_dict()
            # Delete files
            def delete_if_exists(url):
                if not url: return
                path = url.replace('/api/serve_image/', '')
                if path.startswith('/'): path = path[1:]
                local_path = os.path.abspath(path)
                if os.path.exists(local_path):
                    os.remove(local_path)
            
            delete_if_exists(data.get('originalImageUrl'))
            delete_if_exists(data.get('enhancedImageUrl'))
            delete_if_exists(data.get('segmentedImageUrl'))
            
            # Delete doc
            doc.reference.delete()
            count += 1
            
        log_audit_event(uid, "DELETE_MY_DATA", details=f"Deleted {count} logs")
        
        return jsonify({'success': True, 'message': f'Successfully deleted {count} records.'})
        
    except Exception as e:
        print(f"Error deleting data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    session.clear() 
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)