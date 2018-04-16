from django.db import models
from datetime import datetime

# Create your models here.
class DocUsers(models.Model):
    username = models.CharField(max_length=50)
    password = models.CharField(max_length=50)
    email = models.CharField(max_length=50)
    phone = models.CharField(max_length=20)
    doctor_id = models.IntegerField()
    createdAt = models.DateTimeField(default=datetime.now, blank=True)

class PaUsers(models.Model):
    username = models.CharField(max_length=50)
    password = models.CharField(max_length=50)
    email = models.CharField(max_length=50)
    phone = models.CharField(max_length=20)
    patient_id = models.IntegerField()
    createdAt = models.DateTimeField(default=datetime.now, blank=True)   
    
class Results(models.Model):
    patient_id = models.IntegerField()
    doctor_id = models.IntegerField()
    img_loc = models.ImageField()
    cdr = models.FloatField()
    severity = models.FloatField()
    createdAt = models.DateTimeField(default=datetime.now, blank=True) 
    
    
class Imgarch(models.Model):
   paname = models.CharField(max_length = 50)
   img_loc = models.ImageField(upload_to = 'static/pics')
   createdAt = models.DateTimeField(default=datetime.now, blank=True) 

   class Meta:
      db_table = "Imgarch"    