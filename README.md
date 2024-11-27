# Приложение для сбора статистики усталости глаз
- Автор: Базунов Андрей ПИбд42 УлГТУ

Запуск приложения 

1. Создание виртуальной среды
```
python3.13 -m venv venv 
```
2. Активация

- **windows:**
```
./venv/Scripts/activate
```
- **linux:**
```
source ./venv/bin/activate
```

3. Установка зависимостей
```
pip install -r requirements.txt
```

4. Запуск
```
python main.py
```

5. Поделиться данными из папки csv внутри проекта
- Файлы не содержат персональные данные, только вычисленные параметры по алгоритму [EAR](https://ijarsct.co.in/Paper7843.pdf)
- Программа не нагружает ваш компьютер, посколько вычисления проходят по оптимизированным алгоритмам
- Вы можете закрыть программу в любое время, данные автоматически сохраняться в корневую папку проекта с указанием индекса даты