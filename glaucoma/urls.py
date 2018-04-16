from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$',views.index, name= 'index'),
    url(r'profile/',views.profile, name= 'profile'),
    url(r'^saved/', views.SaveProfile, name = 'saved'),
    url(r'^ml/', views.ml, name = 'ml'),
    url(r'^imgp/', views.imgprocess, name = 'imgp'),
    url(r'^signup/', views.signup, name = 'signup'),
    url(r'^dash/', views.dash, name = 'dash'),
    url(r'dashboard/',views.dashboard, name= 'dashboard'),
];