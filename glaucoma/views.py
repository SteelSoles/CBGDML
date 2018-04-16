from django.shortcuts import render
from django.http import HttpResponse,JsonResponse 
from django.core import serializers

from . import forms
from . import models
# Create your views here.
def index(request):
#    return HttpResponse('hey there' )
 return render(request, 'htmlfiles/index1.html' )

def dashboard(request):
    return render(request, 'htmlfiles/dashboard.html')

def dash(request):
    return render(request, 'htmlfiles/dash.html')

def profile(request):
    return render(request, 'htmlfiles/profile.html' )

def signup(request):
#    if request.method == "POST":
        imgarch = models.Results.objects.all()
#    return JsonResponse(imgarch)
        imgarch_serialized = serializers.serialize('json', imgarch)
        return JsonResponse(imgarch_serialized, safe=False) 
#    else:
#        return HttpResponse('hey there')
def ml(cdr):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn import datasets, linear_model
#    import os
#    import datetime
#    t=datetime.date.today()
#    if not os.path.exists('static/pics/'+t.strftime('%Y-%m-%d')):
#        os.makedirs('static/pics/'+t.strftime('%Y-%m-%d'))

    ds= pd.read_csv('GDS.csv')
    X = ds.iloc[:,1].values
    Y = ds.iloc[:,0].values

    from sklearn.cross_validation import train_test_split
    X_train,X_test,Y_train,Y_test = train_test_split(X, Y,train_size = 0.7 ,random_state = 0)

    from sklearn. linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train.reshape(-1,1), Y_train.reshape(-1,1))

#    inp = 3.6
    Y_pred = regressor.predict (cdr)
    result = [(Y_pred[0][0],cdr)]
    df = pd.DataFrame.from_records(result)
    with open('GDS.csv', 'a') as f:
        df.to_csv(f,header=False, index=False)
    # print(df)
    # print(result)
    # print(Y_pred[0][0])
    # print(ds)
    return (Y_pred[0][0])

def imgprocess(adr):
    from PIL import Image
    import numpy as np
    import cv2

    img = Image.open(adr)
    def cup(img):
        gray = img.convert('L')
        height = img.height
        bw = gray.point(lambda x:0 if x<160  else 255,'1')
        bw.save("static/pics/result_cup.jpg")
        img1 = cv2.imread('static/pics/result_cup.jpg')
        x = int(height / 2)
        y = 0
        width = img.width;
        y1 = width - 1;
        b = [255]
        r = img1[x, y]
        r1 = img1[x,y1]
        while (np.any(b != img1[x, y])):
            y = y + 1
            r = img1[x, y]
        while (np.any(b != img1[x, y1])):
            y1 = y1 - 1
            r1 = img1[x, y1]
        img = cv2.imread("static/pics/gp.jpg")
        roi = img[0:height, y:y1]
        cv2.imwrite('static/pics/cup.jpg', roi)
    def disc(img):
        gray = img.convert('L')
        height = img.height
        bw = gray.point(lambda x:0 if x<125 else 255,'1')
        bw.save("static/pics/result_disc.jpg")
        img1 = cv2.imread('static/pics/result_disc.jpg')
        x = int(height / 2)
        y = 0
        width = img.width;
        y1 = width - 1;
        b = [255]
        r = img1[x, y]
        r1= img1[x,y1]
        while (np.any(b != img1[x, y])):
            y = y + 1
            r = img1[x, y]
        while (np.any(b != img1[x, y1])):
            y1 = y1 - 1
            r1 = img1[x, y1]
        img = cv2.imread("static/pics/gp.jpg")
        roi1 = img[0:height, y:y1]
        cv2.imwrite('static/pics/disc.jpg', roi1)

    cup(img)
    disc(img)
    p_img = Image.open("static/pics/cup.jpg")
    p_img1 = Image.open("static/pics/disc.jpg")
    w1 = float(p_img.width)
    w2 = float(p_img1.width)
    ratio = round(w1/w2,2);
    #print "CDR =",ratio
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (ratio)
    


def SaveProfile(request):
   saved = False
   
   if request.method == "POST":
      #Get the posted form
      MyProfileForm = forms.ProfileForm(request.POST, request.FILES)
      
      if MyProfileForm.is_valid():
         imgarch = models.Imgarch()
         imgarch.paname = MyProfileForm.cleaned_data["name"]
         imgarch.img_loc = MyProfileForm.cleaned_data["picture"]
         imgarch.save(imgarch.img_loc)
#         print(imgarch.img_loc) 
         saved = True
   else:
      MyProfileForm = Profileform()
   
   addr =str(imgarch.img_loc).split('/',1)[-1]
   cdr = imgprocess(str(imgarch.img_loc))
   severe = ml(cdr)
   ti = imgarch.createdAt 
   rlt = models.Results()
   rlt.patient_id = 1001
   rlt.doctor_id = 1001
   rlt.img_loc = imgarch.img_loc
   rlt.cdr = cdr
   rlt.severity = severe
   rlt.save()
    
   return render(request, 'htmlfiles/dashboard.html', locals())