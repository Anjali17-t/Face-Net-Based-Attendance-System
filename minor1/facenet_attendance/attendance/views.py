import os
import pickle
import io
import json
import base64
from PIL import Image
import torch
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from .models import Attendance

# Initialize Face Detection & Recognition models
mtcnn = MTCNN(keep_all=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Embeddings file path
pkl_path = os.path.join(settings.BASE_DIR, 'attendance', 'face_embeddings.pkl')

# Load embeddings when server starts
known_embeddings = []
known_names = []
if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        known_embeddings = np.array([e.numpy().flatten() for e in data['embeddings']])
        known_names = data['names']
    print(f"✅ Loaded {len(known_embeddings)} known embeddings.")
else:
    print("ℹ️ No embeddings found yet.")

def index(request):
    return render(request, 'attendance.html')

@csrf_exempt
def mark_attendance(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            image_data = body['image'].split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
            face = mtcnn(image)

            if face is None:
                return JsonResponse({'message': '❌ No face detected'})

            with torch.no_grad():
                embedding = resnet(face.unsqueeze(0)).numpy().flatten()

            if len(known_embeddings) == 0:
                return JsonResponse({'message': '❌ No known faces in the system'})

            distances = np.linalg.norm(known_embeddings - embedding, axis=1)
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]

            if min_distance < 0.7:
                name = known_names[min_idx]
                Attendance.objects.create(name=name)
                return JsonResponse({'message': f'✅ Attendance marked for {name}'})
            else:
                return JsonResponse({'message': '❌ Face not recognized'})
        except Exception as e:
            return JsonResponse({'message': f'❌ Error: {str(e)}'})
    else:
        return JsonResponse({'message': '❌ Invalid request method'})

def upload_new_face(request):
    message = ''
    if request.method == 'POST' and request.FILES.get('image'):
        raw_name = request.POST.get('name', '').strip()
        if not raw_name:
            message = "❌ Name is required."
            return render(request, 'upload_face.html', {'message': message})

        name = raw_name.replace('_', ' ').replace('-', ' ').title()
        img_file = request.FILES['image']
        try:
            img = Image.open(img_file).convert('RGB')
            face = mtcnn(img)
            if face is None:
                message = "❌ No face detected in the image."
            else:
                with torch.no_grad():
                    embedding = resnet(face.unsqueeze(0)).numpy().flatten()

                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                        known_embeddings_local = [e.numpy().flatten() for e in data['embeddings']]
                        known_names_local = data['names']
                else:
                    known_embeddings_local = []
                    known_names_local = []

                known_embeddings_local.append(embedding)
                known_names_local.append(name)

                embeddings_tensors = [torch.tensor(e) for e in known_embeddings_local]
                new_data = {'embeddings': embeddings_tensors, 'names': known_names_local}
                with open(pkl_path, 'wb') as f:
                    pickle.dump(new_data, f)

                message = f"✅ Successfully added and saved embedding for {name}."
        except Exception as e:
            message = f"🔥 Error processing image: {str(e)}"
    return render(request, 'upload_face.html', {'message': message})

def attendance_list(request):
    records = Attendance.objects.all().order_by('-timestamp')
    return render(request, 'attendance.html', {'records': records})

# 🔁 Rebuild embeddings from gallery/
def rebuild_embeddings_from_gallery():
    gallery_path = os.path.join(settings.BASE_DIR, 'photos')

    if not os.path.exists(gallery_path):
     raise FileNotFoundError(f"📁 Folder not found: {gallery_path}")

    embeddings = []
    names = []

    for filename in os.listdir(gallery_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title()
            img_path = os.path.join(gallery_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                face = mtcnn(img)
                if face is not None:
                    with torch.no_grad():
                        emb = resnet(face.unsqueeze(0)).squeeze().numpy()
                        embeddings.append(torch.tensor(emb))
                        names.append(name)
                    print(f"✅ Processed: {filename} as {name}")
                    print(f"[{len(embeddings)}] Processed {filename} → {name}")
                else:
                    print(f"⚠️ No face detected in {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

    with open(pkl_path, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'names': names}, f)

    return len(embeddings), names

# 🔘 View to trigger embedding rebuild
def rebuild_embeddings_view(request):
    if request.method == 'POST':
        count, names = rebuild_embeddings_from_gallery()
        return JsonResponse({'message': f'✅ Rebuilt embeddings for {count} faces.', 'names': names})
    else:
        return render(request, 'rebuild.html')  # Template with a "Rebuild" button
