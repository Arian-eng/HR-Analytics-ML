# نتایج بازتولیدپذیر فصل چهارم

این صفحه نسخه GitHub نتایج فصل چهارم پایان‌نامه است. همه اعداد از اجرای کامل `python run_all.py` روی چهار مجموعه‌داده مستقل تولید شده‌اند؛ بنابراین جدول‌ها و نمودارها دستی وارد نشده‌اند و با اجرای مجدد کد قابل بازتولید هستند.

## طرح تحلیل و داده‌ها

| مجموعه‌داده | تعداد رکورد | ستون خام | متغیر هدف | نوع مسئله | سهم کلاس مثبت |
| --- | --- | --- | --- | --- | --- |
| IBM HR Attrition | 1470 | 35 | Attrition | Binary classification | 16.12% |
| Job Change | 19158 | 14 | target | Binary classification | 24.93% |
| Employee Promotion | 54808 | 14 | is_promoted | Binary classification | 8.52% |
| GHRM–Environmental Performance | 320 | 33 | FEP (derived mean score) | Regression | — |

سه مجموعه‌داده عمومی منابع انسانی برای طبقه‌بندی دودویی و خوشه‌بندی اکتشافی استفاده شدند. مجموعه‌داده GHRM برای رگرسیون `FEP` و خوشه‌بندی مستقل ابعاد مستقیم مدیریت منابع انسانی سبز استفاده شد. رکوردهای چهار منبع با یکدیگر ادغام نشده‌اند.

کنترل کیفیت داده، نبود ردیف تکراری کامل، نبود هدف گمشده و نبود شناسه تکراری یا گمشده در سه مجموعه‌داده دارای شناسه را تأیید کرد. مهم‌ترین محدودیت داده‌ای، گمشده‌بودن `company_type` (۳۲٫۰۵٪)، `company_size` (۳۱٫۰۰٪)، `gender` (۲۳٫۵۳٪) و `major_discipline` (۱۴٫۶۸٪) در Job Change است. جایگذاری داده‌های گمشده داخل هر fold آموزشی انجام می‌شود.

![سهم کلاس مثبت](figures/class_balance.png)

## نتایج طبقه‌بندی

تمام Precision، Recall و F1 جدول زیر مربوط به کلاس مثبت هستند. این نکته برای IBM و Employee Promotion ضروری است، زیرا سهم کلاس مثبت به‌ترتیب ۱۶٫۱۲٪ و ۸٫۵۲٪ است و Accuracy به‌تنهایی می‌تواند گمراه‌کننده باشد.

| مجموعه‌داده | مدل | Accuracy | Precision مثبت | Recall مثبت | F1 مثبت | بهترین F1 در CV |
| --- | --- | --- | --- | --- | --- | --- |
| IBM HR Attrition | Random Forest | 0.8401 | 0.5000 | 0.1277 | 0.2034 | 0.3384 |
| IBM HR Attrition | Decision Tree | 0.7857 | 0.3788 | 0.5319 | 0.4425 | 0.3653 |
| IBM HR Attrition | Linear SVM | 0.7517 | 0.3488 | 0.6383 | 0.4511 | 0.4862 |
| IBM HR Attrition | MLP | 0.8673 | 0.6176 | 0.4468 | 0.5185 | 0.5489 |
| Job Change | Random Forest | 0.7630 | 0.5177 | 0.7194 | 0.6021 | 0.5770 |
| Job Change | Decision Tree | 0.7349 | 0.4793 | 0.7403 | 0.5819 | 0.5642 |
| Job Change | Linear SVM | 0.7437 | 0.4906 | 0.7361 | 0.5888 | 0.5712 |
| Job Change | MLP | 0.7868 | 0.5915 | 0.4670 | 0.5219 | 0.4890 |
| Employee Promotion | Random Forest | 0.9282 | 0.6467 | 0.3469 | 0.4516 | 0.4477 |
| Employee Promotion | Decision Tree | 0.8977 | 0.4090 | 0.4497 | 0.4284 | 0.4372 |
| Employee Promotion | Linear SVM | 0.7584 | 0.2387 | 0.8383 | 0.3716 | 0.3694 |
| Employee Promotion | MLP | 0.9423 | 0.9218 | 0.3533 | 0.5108 | 0.4983 |

در IBM، مدل MLP بالاترین F1 کلاس مثبت را با مقدار **۰٫۵۱۸۵** به دست آورد، در حالی که Linear SVM بیشترین Recall را با مقدار **۰٫۶۳۸۳** ثبت کرد. Random Forest با وجود Accuracy برابر ۰٫۸۴۰۱، F1 مثبت ۰٫۲۰۳۴ داشت و نمونه روشنی از محدودیت Accuracy در داده نامتوازن است.

در Job Change، Random Forest بالاترین F1 مثبت را با مقدار **۰٫۶۰۲۱** ثبت کرد. Decision Tree بیشترین Recall را با مقدار ۰٫۷۴۰۳ داشت، اما Precision پایین‌تر آن باعث شد F1 آن به ۰٫۵۸۱۹ برسد. MLP بیشترین Accuracy را با مقدار ۰٫۷۸۶۸ ثبت کرد، ولی F1 مثبت آن ۰٫۵۲۱۹ بود.

در Employee Promotion، MLP بالاترین F1 مثبت را با مقدار **۰٫۵۱۰۸** و Precision برابر ۰٫۹۲۱۸ به دست آورد. Linear SVM بیشترین Recall را با مقدار **۰٫۸۳۸۳** ثبت کرد، اما Precision آن فقط ۰٫۲۳۸۷ بود؛ بنابراین انتخاب مدل به هزینه خطاهای مثبت کاذب و منفی کاذب وابسته است.

![مقایسه F1 مدل‌های طبقه‌بندی](figures/classification_f1.png)

![ماتریس‌های درهم‌ریختگی](figures/confusion_matrices.png)

## آزمون McNemar

| مجموعه‌داده | مدل A | مدل B | b01 | b10 | p-value |
| --- | --- | --- | --- | --- | --- |
| IBM HR Attrition | Random Forest | Decision Tree | 37 | 21 | 0.0489 |
| IBM HR Attrition | Random Forest | Linear SVM | 52 | 26 | 0.0046 |
| IBM HR Attrition | Random Forest | MLP | 11 | 19 | 0.2012 |
| Job Change | Random Forest | Decision Tree | 211 | 103 | <0.0001 |
| Job Change | Random Forest | Linear SVM | 214 | 140 | 0.0001 |
| Job Change | Random Forest | MLP | 244 | 335 | 0.0002 |
| Employee Promotion | Random Forest | Decision Tree | 543 | 209 | <0.0001 |
| Employee Promotion | Random Forest | Linear SVM | 2347 | 486 | <0.0001 |
| Employee Promotion | Random Forest | MLP | 67 | 222 | <0.0001 |

آزمون McNemar تفاوت الگوی درست/نادرست بودن پیش‌بینی‌های جفت‌شده را می‌سنجد و مستقیماً اختلاف F1 را آزمون نمی‌کند. در IBM، تفاوت Random Forest و MLP در سطح ۰٫۰۵ معنادار نبود (`p=0.2012`)، اما سایر مقایسه‌های گزارش‌شده معنادار بودند.

## نتایج رگرسیون GHRM

در نسخه پایه، `GRS`، `GTD`، `GPA` و `GCM` پیش‌بین‌های `FEP` هستند. در نسخه `GEE+`، متغیر `GEE` نیز اضافه می‌شود.

| نسخه | مدل | R² | MAE | RMSE | CV RMSE |
| --- | --- | --- | --- | --- | --- |
| Base | Random Forest Regressor | 0.3798 | 0.3938 | 0.5114 | 0.5563 |
| Base | Decision Tree Regressor | 0.3698 | 0.3942 | 0.5155 | 0.5892 |
| Base | LinearSVR | 0.4171 | 0.3859 | 0.4958 | 0.5312 |
| Base | MLPRegressor | 0.3534 | 0.4172 | 0.5222 | 0.5928 |
| GEE+ | Random Forest Regressor | 0.3868 | 0.3939 | 0.5085 | 0.5556 |
| GEE+ | Decision Tree Regressor | 0.4134 | 0.3848 | 0.4973 | 0.6073 |
| GEE+ | LinearSVR | 0.4178 | 0.3847 | 0.4955 | 0.5350 |
| GEE+ | MLPRegressor | 0.3712 | 0.3967 | 0.5149 | 0.5700 |

LinearSVR در هر دو نسخه بهترین عملکرد مجموعه آزمون را داشت. در نسخه پایه، R² برابر **۰٫۴۱۷۱** و RMSE برابر **۰٫۴۹۵۸** بود. با افزودن GEE، R² به **۰٫۴۱۷۸** و RMSE به **۰٫۴۹۵۵** رسید. افزایش R² فقط حدود ۰٫۰۰۰۷ است؛ بنابراین این نتیجه شواهدی برای اثر علّی یا میانجی‌گری GEE محسوب نمی‌شود.

![مقایسه RMSE مدل‌های رگرسیون](figures/ghrm_regression_rmse.png)

![مقادیر واقعی و پیش‌بینی‌شده FEP](figures/ghrm_actual_vs_predicted.png)

## نتایج خوشه‌بندی K-Means

متغیر هدف و شناسه‌ها در خوشه‌بندی سه مجموعه‌داده عمومی وارد نشده‌اند. در GHRM نیز `FEP` از تشکیل خوشه‌ها کنار گذاشته شد. برای Job Change و Employee Promotion، Silhouette روی نمونه قطعی ۵٬۰۰۰ رکوردی با `random_state=42` محاسبه شد؛ SSE و Davies–Bouldin از کل داده استفاده می‌کنند.

| مجموعه‌داده | k منتخب | Silhouette | Davies–Bouldin | n برای Silhouette | تفسیر |
| --- | --- | --- | --- | --- | --- |
| IBM HR Attrition | 2 | 0.1313 | 2.6054 | 1470 | تفکیک ضعیف |
| Job Change | 2 | 0.1513 | 2.2778 | 5000 | تفکیک ضعیف |
| Employee Promotion | 5 | 0.1473 | 1.7902 | 5000 | تفکیک ضعیف |
| GHRM–Environmental Performance | 2 | 0.4363 | 0.8611 | 320 | تفکیک اکتشافی متوسط |

مقادیر Silhouette سه مجموعه‌داده عمومی پایین است و جدایی طبیعی خوشه‌ها را ضعیف نشان می‌دهد؛ بنابراین خوشه‌های آن‌ها نباید به‌صورت «سبز/غیرسبز» یا گروه‌های قطعی کارکنان نام‌گذاری شوند. ساختار دوخوشه‌ای GHRM با Silhouette برابر ۰٫۴۳۶۳ و Davies–Bouldin برابر ۰٫۸۶۱۱ تفکیک اکتشافی متوسطی دارد.

![کیفیت خوشه‌بندی در k منتخب](figures/kmeans_selected_quality.png)

### پروفایل توصیفی خوشه‌های GHRM

| خوشه | تعداد | میانگین FEP | انحراف معیار FEP |
| --- | --- | --- | --- |
| 0 | 198 | 3.8510 | 0.5472 |
| 1 | 122 | 3.0410 | 0.7037 |

خوشه ۰ شامل ۱۹۸ مشاهده با میانگین FEP برابر ۳٫۸۵۱۰ و خوشه ۱ شامل ۱۲۲ مشاهده با میانگین FEP برابر ۳٫۰۴۱۰ است. چون FEP در تشکیل خوشه‌ها وارد نشده، این مقایسه یک پروفایل توصیفی پس از خوشه‌بندی است و اثر علّی را نشان نمی‌دهد.

![پروفایل FEP خوشه‌های GHRM](figures/ghrm_cluster_fep.png)

نمودارهای تشخیصی کامل انتخاب k:

- [IBM HR](figures/kmeans_ibm_diagnostics.png)
- [Job Change](figures/kmeans_job_change_diagnostics.png)
- [Employee Promotion](figures/kmeans_promotion_diagnostics.png)
- [GHRM](figures/kmeans_ghrm_diagnostics.png)

## جمع‌بندی فصل

یک الگوریتم واحد در همه مسائل بهترین نبود. MLP بر اساس F1 مثبت در IBM و Employee Promotion برتر بود، در حالی که Random Forest در Job Change بالاترین F1 مثبت را ثبت کرد. در GHRM، LinearSVR بهترین نتیجه مجموعه آزمون را به دست آورد و افزودن GEE بهبود بسیار محدودی ایجاد کرد. خوشه‌بندی سه مجموعه‌داده عمومی ساختار ضعیفی نشان داد و فقط خوشه‌بندی GHRM به سطح تفکیک اکتشافی متوسط رسید.

## محدودیت‌های الزامی در تفسیر

1. نتایج ماهیت پیش‌بینانه یا اکتشافی دارند و رابطه علّی یا میانجی‌گری را اثبات نمی‌کنند.
2. Accuracy در IBM و Employee Promotion باید همراه Precision، Recall، F1 و ماتریس درهم‌ریختگی گزارش شود.
3. گمشده‌بودن برخی متغیرهای Job Change و روش جایگذاری داخل fold باید افشا شود.
4. Silhouette نمونه‌ای Job Change و Employee Promotion باید با `n=5,000` گزارش شود.
5. سه مجموعه‌داده عمومی، سنجه مستقیم GHRM یا رفتار محیط‌زیستی ندارند و فقط مسائل عمومی منابع انسانی را پوشش می‌دهند.
6. تعمیم نتایج به سازمان، صنعت یا کشور دیگر به اعتبارسنجی بیرونی نیاز دارد.

## بازتولید

```bash
python -m pip install -r requirements.txt
python run_all.py
python -m unittest discover -s tests -v
```

جدول‌های ماشین‌خوان در پوشه [`results/tables`](tables/) و نمودارهای آماده GitHub در پوشه [`results/figures`](figures/) قرار دارند. جزئیات کنترل روش‌شناسی و بازمحاسبه معیارها در [`validation/validation_report.md`](../validation/validation_report.md) ثبت شده است.
