# student-performance-predictor


# Student Performance Predictor (MLflow & FastAPI)

ეს პროექტი წარმოადგენს მანქანური სწავლების სრულ ციკლს, რომლის მიზანია სტუდენტის საბოლოო ქულის (0-100) პროგნოზირება მისი სწავლის საათების, დასწრების პროცენტულობის, შესრულებული დავალებებისა და შუალედური გამოცდის ქულის მიხედვით.

## პროექტის სტრუქტურა
* `train.py` - მოდელის გაწვრთნა, ექსპერიმენტის შექმნა და MLflow Tracking (Parameters, Metrics, Model Logging).
* `app.py` - FastAPI სერვისი (Model Serving) მოდელის ავტომატური ჩატვირთვით MLflow-დან და `/predict` ენდფოინთით.
* `requirements.txt` - საჭირო ბიბლიოთეკები.
* `student_performance.ipynb` - კვლევის და მონაცემთა ანალიზის (EDA) ფაილი Google Colab-იდან.

## გაშვების ინსტრუქცია (ლოკალურად)

1. დააინსტალირეთ საჭირო ბიბლიოთეკები:
```bash
pip install -r requirements.txt
