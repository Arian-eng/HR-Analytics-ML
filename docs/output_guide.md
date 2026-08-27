# راهنمای خروجی‌ها

این راهنما برای پیدا کردن پاسخ هر سؤال فنی در مخزن است.

## شناسنامه و ساختار داده

| سؤال | فایل |
|---|---|
| هر دیتاست چند ردیف و ستون دارد؟ | `results/data/dataset_inventory.csv` |
| هش فایل و تعداد ردیف تکراری چیست؟ | `results/data/data_quality.json` |
| نام، نوع، داده گمشده و تعداد مقادیر متمایز هر فیلد چیست؟ | `results/data/column_dictionary.csv` |
| کدام ردیف آموزش یا آزمون بوده؟ | `results/splits/<dataset>.csv` |

## خروجی مدل‌های طبقه‌بندی

مسیر هر مدل به شکل زیر است:

```text
results/classification/<dataset>/<model>/
```

| فایل | محتوا |
|---|---|
| `tuning_results.csv` | تمام ترکیب‌های فراپارامتر و امتیاز CV |
| `model_diagnostics.json` | مدل منتخب، فراپارامترها و پیچیدگی |
| `test_predictions.csv` | شناسه رکورد، مقدار واقعی، پیش‌بینی و درست/غلط بودن |
| `confusion_matrix.csv` | TN، FP، FN و TP |
| `permutation_importance.csv` | تغییر F1 پس از برهم‌زدن هر ویژگی |
| `tree_nodes.csv` | تمام گره‌های Decision Tree و احتمال کلاس مثبت |
| `tree_rules.txt` | قواعد متنی کامل Decision Tree |
| `tree_top_levels.png` | چهار سطح اول درخت برای خوانایی |
| `forest_tree_summary.csv` | عمق و تعداد برگ همه درخت‌های Random Forest |
| `representative_tree_nodes.csv` | تمام گره‌های درخت شماره صفر جنگل |
| `coefficients.csv` | ضرایب Linear SVM |

جدول خلاصه تمام مدل‌ها در `results/tables/classification_metrics.csv` و مقایسه‌های McNemar در `results/tables/mcnemar.csv` است.

## خروجی رگرسیون GHRM

مسیرها:

```text
results/regression/base/<model>/
results/regression/gee/<model>/
```

فایل‌های تنظیم، پیش‌بینی، اهمیت و تشخیص مدل مشابه طبقه‌بندی‌اند. `repeated_cv_metrics.csv` نیز عملکرد هر یک از ۲۵ Fold اعتبارسنجی تکرارشونده را نشان می‌دهد.

`convergence_warnings.json` تعداد هشدارهای همگرایی در تنظیم و ۲۵ Fold تکرارشونده را ثبت می‌کند.

| سؤال | فایل |
|---|---|
| R²، RMSE، MAE و بازه اطمینان چقدر است؟ | `results/tables/regression_metrics.csv` |
| افزودن GEE چه تغییری ایجاد کرده؟ | `results/tables/base_vs_gee_plus.csv` |
| مهم‌ترین متغیرهای هر مدل چیست؟ | `results/tables/regression_feature_importance.csv` |
| معیارها چگونه بدون انتشار ردیف‌ها بازحساب می‌شوند؟ | `results/tables/regression_validation_aggregates.csv` |

## خروجی الگوهای پنهان

برای هر دیتاست:

```text
results/clustering/<dataset>/
```

| فایل | محتوا |
|---|---|
| `kmeans_metrics.csv` | SSE، Silhouette و Davies–Bouldin برای k=2..7 |
| `cluster_assignments.csv` | شماره خوشه هر ردیف |
| `cluster_sizes.csv` | تعداد و سهم اعضای هر خوشه |
| `cluster_distinguishing_features.csv` | ویژگی‌های متمایزکننده و جهت اختلاف |
| `cluster_profile_numeric.csv` | میانگین، میانه و انحراف معیار عددی |
| `cluster_profile_categorical.csv` | مقدار غالب و سهم متغیرهای طبقه‌ای |
| `cluster_target_summary.csv` | توزیع/میانگین هدف پس از خوشه‌بندی |

در GHRM، ارتباط الگوهای سبز با عملکرد محیط‌زیستی دقیقاً از کنار هم گذاشتن `cluster_distinguishing_features.csv` و `cluster_target_summary.csv` دیده می‌شود. پنج سازه سبز خوشه را می‌سازند و میانگین `FEP` بعداً برای هر خوشه گزارش می‌شود.

پروفایل‌های تفصیلی سه دیتاست عمومی کارکنان فقط در اجرای محلی تولید می‌شوند. مخزن عمومی برای آن‌ها معیار انتخاب k و اندازه خوشه را نگه می‌دارد و آمار درآمد/نرخ، شناسه و پروفایل فردی را منتشر نمی‌کند. خروجی تفصیلی GHRM در مخزن باقی می‌ماند، چون در سطح سازه‌های پژوهش و بدون شناسه پاسخ‌دهنده گزارش شده است.

## جدول‌ها و شکل‌های آماده استفاده

جدول‌های تجمیعی در `results/tables/` و نمودارهای نهایی در `results/figures/` هستند. `results/analysis_report_fa.md` همه نتیجه‌های مهم و پیوند فایل جزئیات را در یک گزارش جمع می‌کند.

## ردگیری یک عدد نمونه

برای بررسی یک F1 مانند عدد مدل MLP در IBM:

1. ردیف آزمون در `results/splits/ibm.csv` مشخص است.
2. مقدار واقعی و پیش‌بینی در `results/classification/ibm/mlp/test_predictions.csv` است.
3. F1 از همان دو ستون دوباره محاسبه می‌شود.
4. نتیجه در `results/classification/classification_metrics.csv` و جدول تجمیعی تکرار می‌شود.
5. `scripts/validate_results.py` برابری این مقادیر را کنترل می‌کند.

دو فایل مرحله ۱ و ۲ فقط در اجرای محلی ساخته می‌شوند و به دلیل حفاظت از خروجی‌های فردی HR در GitHub عمومی قرار نمی‌گیرند. در نسخه عمومی، همان F1 مستقلاً از چهار مقدار ماتریس درهم‌ریختگی بازحساب می‌شود. برای رگرسیون نیز `regression_validation_aggregates.csv` شامل تعداد رکورد و مجموع خطاهای مربع/مطلق است، نه پیش‌بینی فردی.

برای عددهایی مانند ۰٫۴۶۵ نیز باید ابتدا نام معیار و فایل منبع مشخص شود. در خروجی فعلی هیچ عدد بدون نام ستون، مدل، دیتاست و مسیر محاسبه منتشر نمی‌شود.
