from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # 🌟 loan_app ထဲက urls.py နှင့် ချိတ်ဆက်ပေးထားခြင်း (အကောင်းဆုံးစနစ်)
    path('api/', include('loan_app.urls')),
]