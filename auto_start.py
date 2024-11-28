import os
import subprocess
import sys


def is_virtualenv_active(venv_dir):
    """
    Проверяет, активирована ли виртуальная среда.
    """
    return os.path.abspath(sys.executable).startswith(os.path.abspath(venv_dir))


def create_virtualenv(venv_dir):
    """
    Создает виртуальную среду, если она не существует.
    """
    if not os.path.exists(venv_dir):
        print("Создаётся виртуальная среда...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        print("Виртуальная среда создана.")
    else:
        print("Виртуальная среда уже существует.")


def are_requirements_installed(requirements_file):
    """
    Проверяет, установлены ли все зависимости из requirements.txt.
    """
    if not os.path.exists(requirements_file):
        print(f"Файл {requirements_file} не найден.")
        return True  # Если файла нет, считаем, что установка не требуется

    try:
        with open(requirements_file, "r") as file:
            packages = [line.strip() for line in file if line.strip() and not line.startswith("#")]

        print("Проверяем установленные пакеты...")
        for package in packages:
            package_name = package.split("==")[0]  # Убираем версию, если указана
            result = subprocess.run([sys.executable, "-m", "pip", "show", package_name], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Пакет {package_name} не установлен.")
                return False
        print("Все пакеты установлены.")
        return True
    except Exception as e:
        print(f"Ошибка при проверке зависимостей: {e}")
        return False


def activate_and_install(requirements_file):
    if are_requirements_installed(requirements_file):
        print("Зависимости уже установлены. Пропускаем установку.")
        return

    print("Обновляем pip...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--trusted-host", "pypi.org", "--trusted-host",
         "pypi.python.org", "--trusted-host", "files.pythonhosted.org"])

    if os.path.exists(requirements_file):
        print("Устанавливаем зависимости из файла requirements.txt...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            requirements_file,
            "--trusted-host", "pypi.org",
            "--trusted-host", "pypi.python.org",
            "--trusted-host", "files.pythonhosted.org"
        ])
        print("Все зависимости установлены.")
    else:
        print(f"Файл {requirements_file} не найден. Зависимости не установлены.")


def reboot(*reboot_args: str, venv_dir='.venv'):
    python_executable = os.path.join(venv_dir, "bin", "python")
    os.system('cls' if os.name == 'nt' else 'clear')
    os.execv(python_executable, [python_executable] + sys.argv + [*reboot_args])


def setup_environment():
    venv_dir = ".venv"
    requirements_file = "requirements.txt"

    create_virtualenv(venv_dir)

    if not is_virtualenv_active(venv_dir):
        # Перезапускаем приложение с виртуальной средой
        print("Перезапуск приложения с активированной виртуальной средой...")
        reboot(venv_dir=venv_dir)
    else:
        # Устанавливаем зависимости
        activate_and_install(requirements_file)
