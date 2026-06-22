from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # သင့်ရဲ့ loan_app API တွေနဲ့ ချိတ်ဆက်ဖို့ လမ်းကြောင်းဖွင့်ပေးထားခြင်း
    path('api/', include('loan_app.urls')),
]