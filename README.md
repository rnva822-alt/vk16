# VK Long Search: локальный длительный поиск

Этот проект продолжает offline-поиск улучшений от подтверждённого лидерборд-решения с F1 **0,47355959**. Он перебирает регуляризацию, силу временного затухания и seed для временно-взвешенного CatBoost, затем подбирает веса трёхкомпонентного ансамбля на последнем временном фолде.

> Каждый завершённый опыт сразу записывается в `results/experiment_log.csv`. Процесс можно остановить и запустить повторно: уже завершённые комбинации не будут обучаться заново.

## Состав проекта

| Путь | Назначение |
|---|---|
| `data/train.csv`, `data/test.csv` | Исходные данные конкурса |
| `long_search.py` | Чекпойнтный движок длительного поиска |
| `results/experiment_log.csv` | Журнал всех завершённых экспериментов |
| `results/leaderboard.csv` | Отсортированная таблица лучших offline-конфигураций |
| `checkpoints/state.json` | Последний успешно записанный эксперимент |
| `solution.py`, `run_experiments.py`, `build_advanced_ensembles.py` | Проверенные модули подготовки признаков и моделей |

## Запуск на Windows

Откройте PowerShell в папке проекта и выполните:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python long_search.py --mode extended --max-hours 10
```

Для запуска в фоне PowerShell:

```powershell
Start-Process python -ArgumentList 'long_search.py --mode extended --max-hours 10' -RedirectStandardOutput 'results\live.log' -RedirectStandardError 'results\live_error.log' -NoNewWindow
```

## Запуск на macOS или Linux

Откройте Terminal в папке проекта и выполните:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
nohup python long_search.py --mode extended --max-hours 10 > results/live.log 2>&1 &
```

Проверить, что поиск идёт, можно командой:

```bash
tail -f results/live.log
```

## Режимы

| Режим | Назначение | Время |
|---|---|---|
| `--mode fast` | Первичная проверка только с основным seed | Короткий прогон |
| `--mode extended` | Полная матрица с несколькими seed | Рекомендуемый длительный прогон |

Параметр `--max-hours 10` ограничивает один запуск десятью часами. Для теста перед длинным запуском можно выполнить:

```bash
python long_search.py --mode fast --max-trials 1
```

## После завершения

Пришлите мне файлы `results/leaderboard.csv`, `results/experiment_log.csv` и `checkpoints/state.json` либо подключите эту папку к задаче. Я отберу лучшие устойчивые конфигурации, обучу финальные модели на полном train и подготовлю новые `submission.csv`.

## Важное ограничение

Offline F1 на временном фолде служит для ранжирования гипотез, но не гарантирует тот же результат на скрытом лидерборде. Сохраняйте уже подтверждённый файл с F1 0,47355959 как текущий лучший до проверки новых посылок.
