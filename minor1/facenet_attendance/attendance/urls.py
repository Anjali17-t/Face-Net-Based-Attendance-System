from django.urls import path
from . import views
from .views import rebuild_embeddings_view

urlpatterns = [
    path('', views.index, name='index'),
    path('mark_attendance/', views.mark_attendance, name='mark_attendance'),
    path('upload/', views.upload_new_face, name='upload_new_face'),
    path('records/', views.attendance_list, name='attendance_list'),
    path('attendance-list/', views.attendance_list, name='attendance_list'),
    path('rebuild_embeddings/', views.rebuild_embeddings_view, name='rebuild_embeddings'),
]
