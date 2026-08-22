# نتایج مرجع فصل چهارم — مطابق پایان‌نامه اصلی

> **مرجع عددی این فایل، پایان‌نامه اصلی `آرین پوراسدedit(2).docx` است.** این صفحه برای همسان‌سازی GitHub با اعداد گزارش‌شده در پایان‌نامه نگهداری می‌شود. اعداد این صفحه نباید به‌عنوان خروجی اجرای دیگری معرفی شوند مگر اینکه همان اجرای دقیق بازتولید شده باشد.

## طرح تحلیل و داده‌ها

پژوهش از چهار مجموعه‌داده مستقل استفاده می‌کند و رکوردها با یکدیگر Merge نشده‌اند. مجموع رکوردهای خام چهار منبع **75,756** است و این عدد به معنی یک نمونه تجمیع‌شده نیست.

| مجموعه‌داده | تعداد رکورد | متغیر هدف | نوع مسئله | سهم کلاس مثبت |
| --- | ---: | --- | --- | ---: |
| IBM HR Analytics Employee Attrition & Performance | 1,470 | `Attrition` | Binary classification | 16.1% (237) |
| HR Analytics: Job Change of Data Scientists | 19,158 | `target` | Binary classification | 24.9% (4,777) |
| Employee Promotion Prediction | 54,808 | `is_promoted` | Binary classification | 8.5% (4,668) |
| GHRM–Environmental Performance | 320 | `FEP` | Regression + exploratory clustering | — |

سه مجموعه‌داده عمومی منابع انسانی برای طبقه‌بندی دودویی و خوشه‌بندی اکتشافی و مجموعه‌داده GHRM برای پیش‌بینی FEP و خوشه‌بندی مستقل استفاده شدند. fileciteturn120file3L118-L127

## طراحی تقسیم داده و تنظیم مدل

برای مسائل طبقه‌بندی، داده‌ها با نسبت **80% آموزش / 20% آزمون** تقسیم شدند و تنظیم فراپارامترها فقط روی مجموعه آموزش انجام شد. اعتبارسنجی متقاطع سه‌بخشی در اجرای نهایی فصل چهارم استفاده شد. برای GHRM نیز 256 مشاهده آموزش و 64 مشاهده آزمون وجود داشت و آزمون بدون stratification انجام شد. fileciteturn125file3L185-L203 fileciteturn125file4L218-L230

## نتایج طبقه‌بندی

تمام Precision، Recall و F1 زیر مربوط به **کلاس مثبت** هستند.

### IBM HR Analytics — Attrition

برای IBM، مجموعه آزمون 294 مشاهده‌ای بود. چهار مدل Random Forest، Decision Tree، LinearSVC و MLPClassifier اجرا شدند. fileciteturn120file4L194-L217

| مدل | Accuracy | Precision مثبت | Recall مثبت | F1 مثبت |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.837 | 0.462 | 0.128 | 0.200 |
| Decision Tree | 0.772 | 0.321 | 0.383 | 0.350 |
| LinearSVC | 0.864 | 0.652 | 0.319 | 0.429 |
| MLPClassifier | **0.874** | **0.708** | 0.362 | **0.479** |

MLPClassifier بهترین توازن کلی را در IBM داشت. آزمون McNemar برای MLPClassifier در برابر LinearSVC مقدار **p=0.508** را نشان داد؛ در برابر Random Forest و Decision Tree تفاوت معنادار گزارش شد (`p<0.001` و `p=0.035`). fileciteturn125file5L242-L257

### Job Change — `target`

در Job Change، 15,326 مشاهده در آموزش و 3,832 مشاهده در آزمون قرار گرفتند؛ آزمون شامل 2,877 نمونه کلاس منفی و 955 نمونه کلاس مثبت بود. fileciteturn125file3L185-L200

| مدل | Accuracy | Precision مثبت | Recall مثبت | F1 مثبت |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.776 | 0.569 | 0.419 | 0.483 |
| Decision Tree | **0.785** | 0.585 | 0.477 | **0.526** |
| LinearSVC | 0.778 | **0.586** | 0.377 | 0.459 |
| MLPClassifier | **0.785** | 0.584 | 0.476 | 0.525 |

Decision Tree بالاترین F1 را با **0.526** به دست آورد و MLPClassifier با F1 برابر **0.525** بسیار نزدیک بود. در متن پایان‌نامه، آزمون McNemar این دو مدل تفاوت معناداری نشان نداد. fileciteturn120file7L301-L322

### Employee Promotion — `is_promoted`

در Promotion، 43,846 مشاهده در آموزش و 10,962 مشاهده در آزمون قرار گرفتند. مجموعه آزمون شامل 10,028 نمونه کلاس منفی و 934 نمونه کلاس مثبت بود. fileciteturn120file2L81-L104

| مدل | Accuracy | Precision مثبت | Recall مثبت | F1 مثبت |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.933 | 0.803 | 0.279 | 0.415 |
| Decision Tree | 0.937 | 0.774 | **0.374** | 0.504 |
| LinearSVC | 0.933 | **0.951** | 0.228 | 0.368 |
| MLPClassifier | **0.942** | 0.932 | 0.350 | **0.509** |

MLPClassifier بهترین F1 را با **0.509** ثبت کرد؛ Decision Tree بیشترین Recall را با **0.374** داشت و LinearSVC بالاترین Precision را با **0.951** ثبت کرد. fileciteturn120file0L12-L35

## نتایج رگرسیون GHRM

در تحلیل پایه، `GRS`، `GTD`، `GPA` و `GCM` برای پیش‌بینی `FEP` استفاده شدند. در تحلیل تکمیلی، `GEE` نیز به ورودی‌ها اضافه شد. چهار مدل Random Forest Regressor، Decision Tree Regressor، LinearSVR و MLPRegressor ارزیابی شدند. fileciteturn121file7L435-L467

### تحلیل پایه

| مدل | R² | RMSE | MAE |
| --- | ---: | ---: | ---: |
| Random Forest Regressor | **0.432** | **0.489** | 0.386 |
| Decision Tree Regressor | 0.422 | 0.494 | **0.381** |
| LinearSVR | 0.417 | 0.496 | 0.386 |
| MLPRegressor | 0.356 | 0.521 | 0.403 |

fileciteturn121file7L448-L467

### تحلیل تکمیلی با افزودن GEE

| مدل | R² | RMSE | MAE |
| --- | ---: | ---: | ---: |
| Random Forest Regressor | 0.425 | 0.492 | 0.385 |
| Decision Tree Regressor | 0.406 | 0.501 | 0.382 |
| LinearSVR | 0.418 | 0.495 | 0.385 |
| MLPRegressor | 0.422 | 0.494 | 0.398 |

fileciteturn121file0L15-L40

## نتایج K-Means

K-Means برای هر چهار مجموعه‌داده مستقل بررسی شد. `k` از 2 تا 7 ارزیابی و تعداد نهایی برای هر مجموعه‌داده جداگانه تعیین شد. هدف و شناسه‌ها وارد تشکیل خوشه‌ها نشدند و در GHRM نیز `FEP` از ورودی خوشه‌بندی کنار گذاشته شد. fileciteturn119file9L401-L428

| مجموعه‌داده | k منتخب | SSE | Silhouette | Davies-Bouldin |
| --- | ---: | ---: | ---: | ---: |
| IBM HR Analytics | 2 | 29253.11 | 0.153 | 2.382 |
| Job Change | 3 | 11219.41 | **0.577** | 0.664 |
| Employee Promotion | 5 | 191896.70 | 0.257 | 1.270 |
| GHRM | 2 | 770.65 | 0.436 | **0.861** |

fileciteturn119file4L175-L198

در Job Change، سه خوشه انتخاب شد و Silhouette برابر **0.577** بود. در Employee Promotion، ساختار پنج‌خوشه‌ای با Silhouette برابر **0.257** انتخاب شد. در GHRM، دو خوشه با Silhouette برابر **0.436** و Davies-Bouldin برابر **0.861** انتخاب شد. fileciteturn119file6L319-L331

## جمع‌بندی نتایج مرجع

- **IBM:** MLPClassifier با Accuracy=0.874 و F1=0.479 بهترین توازن کلی را نشان داد.
- **Job Change:** Decision Tree با F1=0.526 بهترین عملکرد را ثبت کرد و MLPClassifier با F1=0.525 بسیار نزدیک بود.
- **Employee Promotion:** MLPClassifier با Accuracy=0.942 و F1=0.509 بهترین توازن را نشان داد.
- **GHRM:** Random Forest Regressor در تحلیل پایه با R²=0.432 بهترین R² را داشت.
- **K-Means:** kهای منتخب به‌ترتیب IBM=2، Job Change=3، Employee Promotion=5 و GHRM=2 هستند.

این نتایج باید به‌عنوان **اعداد مرجع پایان‌نامه** در GitHub در نظر گرفته شوند؛ هر خروجی جدید حاصل از اجرای کد باید با این جدول تطبیق داده شود و در صورت اختلاف، علت اختلاف مستند شود.