"""
Definition of views.
"""

from datetime import datetime
from email.mime import image
import re
from django.shortcuts import render
from django.http import HttpRequest, JsonResponse
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from django.views.decorators.csrf import csrf_exempt


model = tf.keras.models.load_model(r"D:\\Visual Studio Projects\\DjangoWebProject1\\DjangoWebProject1\\app\\assets\\model.h5")
CLASS_LABELS = [
    "Acne and Rosacea Photos", "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions", "Atopic Dermatitis Photos", "Bullous Disease Photos", "Cellulitis Impetigo and other Bacterial Infections", "Eczema Photos",
    "Exanthermas and Drug Eruptions", "Hair Loss Photos Alopecia and other Hair Diseases", "Herpes HPV and other STDs Photos", "Light Disease and Disorders of Pigmentation", "Lupus and other Connective Tissue diseases", "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease", "Poison Ivy Photos and other Contact Dermatitis", "Psoriasis pictures Lichen Planus and related diseases", "Scabies Lyme Disease and other Infestations and Bites", "Seborrheic Keratoses and other Benign Tumors", "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections", "Urticaria Hives", "Vascular Tumors", "Vasculities Photos", "Warts Molluscum and other Viral Infections" 
    ]

@csrf_exempt
def predict(request):
    assert isinstance(request, HttpRequest)
    if request.method == 'POST':
        try:
            file = request.FILES.get('file')
            if not file:
                return JsonResponse({"error": "No such file or directory"}, status=400)

            image = Image.open(file).resize((224, 224), Image.Resampling.LANCZOS)  # Specify a resampling filter
            image = np.array(image) / 255.0  # Normalize pixel values
            image = np.expand_dims(image, axis=0)

            # Make predictions
            predictions = model.predict(image)[0]
            top_5_indices = np.argsort(predictions)[-5:][::-1]
            top_5_predictions = [
                {"class": CLASS_LABELS[i], "probability": float(predictions[i])}
                for i in top_5_indices
            ]

            return JsonResponse({"predictions": top_5_predictions})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)


def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/index.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )

def contact(request):
    """Renders the contact page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/contact.html',
        {
            'title':'Contact',
            'message':'Your contact page.',
            'year':datetime.now().year,
        }
    )

def about(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/about.html',
        {
            'title':'About',
            'message':'Your application description page.',
            'year':datetime.now().year,
        }
    )
