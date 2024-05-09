import datetime

def get_formatted_time():
    now = datetime.datetime.now()
    return '{YEAR:04d}-{MONTH:02d}-{DAY:02d} {HOUR:02d}:{MINUTE:02d}:{SECOND:02d}'.format(
        YEAR=now.year,
        MONTH=now.month,
        DAY=now.day,
        HOUR=now.hour,
        MINUTE=now.minute,
        SECOND=now.second,
    )

def print_message(message):
    print('{DATE} {MESSAGE}'.format(
        DATE=get_formatted_time(),
        MESSAGE=message,
    ))

def print_title(message):
    print(message)

def print_result(message):
    print(message)

def print_line_break():
    print()

def print_progress(task, progress, status):
    print('{DATE} [{PROGRESS:5.1f}%] {TASK}: {STATUS}'.format(
        DATE=get_formatted_time(),
        TASK=task,
        PROGRESS=progress*100,
        STATUS=status,
    ))
