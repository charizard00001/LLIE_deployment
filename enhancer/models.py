from django.db import models

class UploadedImage(models.Model):
    input_image = models.ImageField(upload_to='input_images/')
    enhanced_image = models.ImageField(upload_to='output_images/', null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

# not needed