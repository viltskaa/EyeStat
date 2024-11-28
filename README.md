# Приложение для сбора статистики усталости глаз
- Автор: Базунов Андрей ПИбд42 УлГТУ

---

## _Важно!_

Перед запуском приложения необходимо убедиться, что в вашей ОС установлен **[CMake](https://cmake.org)**

- для проверки, вы можете выполнить данную команду, чтобы узнать версию утилиты 

```bash
    cmake --version
```

> Windows:
>```
>winget install -e --id Kitware.CMake
>```

>Linux:
>```bash
>  brew install cmake
>```

---
Запуск приложения из корневой папки проекта

```bash
  python3.13 main.py
```

> Программа сама установит все зависимости и перезагрузится, в автоматическом режиме. 
> Вам не надо беспокоится о ее работе 😘

---
## Поделиться данными из папки csv внутри проекта
- Файлы не содержат персональные данные, только вычисленные параметры по алгоритму **[EAR](https://ijarsct.co.in/Paper7843.pdf)**
- Программа не нагружает ваш компьютер, посколько вычисления проходят по оптимизированным алгоритмам
- Вы можете закрыть программу в любое время, данные автоматически сохраняться в **корневую папку проекта** 
с указанием индекса даты и хэша вашего процессора

> Отправить файл можно мне в телеграмм [@Viltskaa](https://t.me/Viltskaa?&text=ear)

## Спасибо за внимание и предоставленные данные! 🥰