from django.urls import path
from base import views
from django.conf.urls.i18n import set_language


urlpatterns = [
    path('', views.home, name='home'),
    path('result/', views.result, name='result'),
    path('set_language/', set_language, name='set_language'),
]