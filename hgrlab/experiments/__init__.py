import datetime
import numpy as np

def run_experiments(
    title,
    experiments,
    assets_dir,
    classifier_names,
    user_ids,
    folds,
    val_size_per_class=None,
    setup=None,
):
    start_ts = datetime.datetime.now()

    print_line_break()
    print_title(title)

    if callable(setup):
        print_line_break()
        setup()

    for i, experiment in enumerate(experiments):
        print_line_break()
        experiment(
            experiment_id=i+1,
            total_experiments=np.size(experiments),
            assets_dir=assets_dir,
            classifier_names=classifier_names,
            user_ids=user_ids,
            folds=folds,
            val_size_per_class=val_size_per_class,
        )

    end_ts = datetime.datetime.now()
    print_line_break()
    print_message('Finished all experiments')
    print_message('Total time elapsed: %s' % str(end_ts - start_ts))
    print_line_break()

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
